#using carCut Conda env
from datetime import datetime
import os
import uuid
from flask import Flask, render_template, session, request,send_file
from werkzeug.utils import secure_filename
from pathlib import Path
from google.cloud import storage
from firebase import firebase
fbdb = firebase.FirebaseApplication('https://infiniteloop.firebaseio.com/', None)
firebase_collection='/carcut_queue/'

# ----------------------- start global variable --------------------------------
# ------------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder='ui/templates',
    static_url_path="/ui/static",
    static_folder='ui/static')
app.config['SECRET_KEY'] = 'SKRTKY'
app.config['UPLOAD_FOLDER'] =  Path.cwd() / 'data'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 * 20  # max dataset size 20 GB 
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'zip'}
# ___________________________ end global variable ______________________________


# ----------------------- start main menue route -------------------------------
# ------------------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    session['PROCESS_LIST'] = 'None'
    # session['CURRENT_TRAIN_ID'] = 'None'
    # session.pop('CURRENT_TRAIN_STATUS', None)
    # if 'CURRENT_DATASET' not in session:
    #     session['CURRENT_DATASET'] = {'dataset_uid':'mvtec_ad','dataset_name':'mvtec_ad'} 
    return render_template('add.html')

@app.route('/addzip', methods=['GET'])
def dataset():
    return render_template('add_zip.html')


@app.route('/addimage', methods=['GET'])
def train():
    return render_template('add_image.html')

# ___________________________ end main menue route ____________________________

@app.route('/api/add_queue', methods=['POST'])
def add_queue():
    print(f"{'*'*80}\nFile Uploaded: {request.files} \n {'*'*80}\n")
    if 'dsFile' not in request.files:
        return 'Failed. No file uploaded.'

    f = request.files['dsFile']

    if f.filename == '':
            return 'Failed. No file was selected.'
    if f and allowed_file(f.filename):
        data=request.form
        print(data)
        queue_id = uuid.uuid4().hex
        status='unprocessed'
        # print(f.filename)
        filename = secure_filename(f.filename)
        print(filename)
        file_extension = f.filename.split('.')[-1]
        print(f"file_extension: {file_extension}")
        uploaded_file_path=app.config['UPLOAD_FOLDER'] / filename
        f.save(uploaded_file_path)
        #Upload zip file to gcp bucket
        url=upload_blob_gcp(uploaded_file_path,f"{status}/{queue_id}.{file_extension}")
        
        #save data to firebase db, which acts as queue
        current_dict={
            'Status': status,
            'Url':url,
            'FileType':file_extension,
            'QueueId' : queue_id,
            'AddTime': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'OutputStyle':data['OutputStyle'],
            'BlurStrength':data['BlurStrength'],
        }
        try:
            current_dict.update({'OutputSize':data['OutputSize']})
        except:
            print("OutputSize not found")
        print(current_dict)
        fbdb.put(firebase_collection,queue_id, data=current_dict)
        session['CURRENT_QUEUE_ID']=queue_id
        os.remove(uploaded_file_path)
        return current_dict
    return "Failed, file type not supported"

@app.route('/api/ai/exterior', methods=['GET'])
def UnprocessedZip():
    creadential=request.headers.get('X-App-Credential')
    # print(creadential)
    if(creadential=='CHOWAGIKEN'):
        res = fbdb.get(firebase_collection, '')
        if not res:
            return {'Status': 'empty','WaitTime':5}
        res= list(res.values())
        res=next((item for item in res if item["Status"] == "unprocessed"), None)
        if not res:
            return {'Status': 'empty','WaitTime':2}
        else:
            res['Status'] = 'processing'
            fbdb.put(firebase_collection,res['QueueId'], data=res)
            return res
    else:
        return {'Status': 'invalid credential'},401

@app.route('/api/ai/exterior', methods=['POST'])
def ProcessedFile():
    creadential=request.headers.get('X-App-Credential')
    data=request.form
    processed_file = request.files['ProcessedFile']
    print(data)
    processed_file_path=app.config['UPLOAD_FOLDER'] / f"processed.{data['FileType']}"
    if(creadential=='CHOWAGIKEN'):
        res = fbdb.get(firebase_collection, data['QueueId'])
        if(res['Status'] == 'processed'):
            return {'Status': 'processed','QueueId' : data['QueueId']}
        if not processed_file:
            return {'Status': 'file upload error'},401
        processed_file.save(processed_file_path)
        # print(type(processed_file_path))
        res['Status'] = 'processed'
        processed_url=upload_blob_gcp(processed_file_path,f"{res['Status']}/{data['QueueId']}.{data['FileType']}")
        print(processed_url)
        current_time=datetime.now()
        res['ProcessedTime'] = current_time.strftime("%d/%m/%Y %H:%M:%S")
        add_time=datetime.strptime(res['AddTime'], "%d/%m/%Y %H:%M:%S")
        process_duration=current_time-add_time
        print(process_duration.total_seconds())
        res['ProcessDuration']=process_duration
        res['ProcessedUrl']=processed_url
        fbdb.put(firebase_collection,res['QueueId'], data=res)
        return {'Status': 'success','QueueId' : data['QueueId']}
    else:
        return {'Status': 'invalid credential'},401

@app.route('/api/check_status/<queue_id>', methods=['GET'])
def CheckProcessStatus(queue_id:str):
    res = fbdb.get(firebase_collection, queue_id)
    print(res)
    print(type(res))
    return {'Status':"success",'data':res}


# ------------------------- start util methods -----------------------------
# --------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_blob_gcp( source_file_name, destination_blob_name,bucket_name='test_server_api'):
    """Uploads a file to the bucket."""
    storage_client = storage.Client.from_service_account_json(
        './gcp_service_key.json')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )
    # #returns a signed url
    # return blob.generate_signed_url(
    #     expiration=dt.timedelta(days=7),
    #     version='v4')
    blob.make_public()
    return blob.public_url

# _____________________________ end util methods ___________________________

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)