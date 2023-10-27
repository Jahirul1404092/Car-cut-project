import json

from cv2 import blur
import config
import os
from pathlib import Path
import shutil
import urllib.request as urllib_req
import requests
import time
from object_detection.detect import run as detect_object
from instance_segmentation.instance_inference import main as instance_segment
from semantic_segmentation.segmentation_inference import main as sementic_segment
from zipfile import ZipFile, ZIP_DEFLATED
from argparse import ArgumentParser



# Global Variable
data_path='./data'
results = dict(); 
results['trasparent_path'] = './data/output/transparent_bg'
results['white_path']   = './data/output/white_bg'
results['blur_path']   = './data/output/blur_bg'
results['zip_out_path']= './data/output.zip'
results['output_path']='./data/output'
zip_path = './data/inputdata.zip' # get *.zip url and place it under `/data/
inputdata_path = './data/inputdata' # will auto-create

def unzip_folder(zip_path, unzip_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_path)

def zip_output(output_type:str = 'blur_path'):
    dir = Path(results[output_type])
    print(dir)
    with ZipFile(results['zip_out_path'], "w", ZIP_DEFLATED) as zip_file:
        for entry in dir.rglob("*"):
            zip_file.write(entry, entry.relative_to(dir))

def run_car_cut(blur_level, output_size,dilation_iteration):
    # initially it was needed.
    # but not removing for future references.
    # detect_object(
    #     weights="./object_detection/yolov5n6.pt",
    #     source="./data/inputdata",
    #     imgsz=[640,640],
    #     crop_output="./data/instance/input",
    #     device='cpu',
    #     save_crop=True,
    #     only_max_area=True,
    #     classes=[2,5,7])
    if(blur_level %2)==0:
        #Blur level have to be a odd number, else there is an error
        blur_level+=1
    if(config.GPU):
        device=''
        print("Running on GPU")
    else:
        device='cpu'
    instance_segment(
        MODEL_WEIGHT_PATH="./instance_segmentation/model_final.pth",
        INPUT_DIRECTORY   = './data/inputdata',
        OUTPUT_DIRECTORY  = './data/semantics/input',
        OUTPUT_DIRECTORY2 = './data/semantics/bigger_mask',
        DEVICE= device, #cpu or keep empty to run on gpu
        BLUR_LEVEL = blur_level
    )
    

    sementic_segment(
        MODEL_WEIGHT_PATH='./semantic_segmentation/model.h5',
        DIRECTORY = './data/semantics/input',
        DIRECTORY2 = './data/inputdata',
        DIRECTORY3 = './data/semantics/bigger_mask',
        DIRECTORY_TRANSPARENT_OUTPUT = results['trasparent_path'],
        DIRECTORY_WHITE_BG_OUTPUT = results['white_path'],
        DIRECTORY_BLUR_BG_OUTPUT = results['blur_path'],
        BLUR_LEVEL=blur_level,
        OUTPUT_SIZE=output_size,
        DILATION_ITERATION=dilation_iteration #Added by jahirul
    )  
    
    return results

def run_process(output_type:str ='blur_path',file_type:str="zip", blur_level:int=43, output_size:str='original',dilation_iteration:int=12):#dilation Added by jahirul):
    #handle image(png,jpeg) file_type 
    if(file_type=="zip"):
        # unzip the zip file
        unzip_folder(zip_path, inputdata_path)
    
    # run the program.
    run_car_cut(blur_level, output_size,dilation_iteration)

    if(file_type=="zip"):
        # make zip file from output of processed images
        zip_output(output_type)

        #Cleanup files after each process loop
        data_cleanup(inputdata_path)
        return results['zip_out_path']
    else :
        if output_type=='trasparent_path':
            image_out_path=f"{data_path}/output.png"
            shutil.copyfile(f"{results[output_type]}/{queue_id}.png", image_out_path)
        else:
            image_out_path=f"{data_path}/output.{file_type}"
            shutil.copyfile(f"{results[output_type]}/{queue_id}.{file_type}", image_out_path)
        #cleanup image files
        data_cleanup(inputdata_path)
        # os.remove(f"{inputdata_path}/{queue_id}.{file_type}")
        return image_out_path

def data_cleanup(inputdata_path):

    #Only Delete folders in inputdata_path
    dir_list=os.listdir(inputdata_path)
    for dir_name in dir_list:
        dir_path=inputdata_path+"/"+dir_name
        if(os.path.isdir(dir_path)):
            shutil.rmtree(dir_path)
    
    #only delete image files in inputdata_path
    file_ext=(".jpeg",".jpg",".png")
    filtered_files = [file for file in os.listdir(inputdata_path) if file.endswith(file_ext)]
    for file in filtered_files:
        os.remove(os.path.join(inputdata_path, file))

    # Remove output dirs
    shutil.rmtree(results['output_path'])
    # Remove semantic dirs
    shutil.rmtree('./data/semantics')

def send_processed_file(output_filename,queue_id,file_type):
    # send image file as file post request along with queue id
    files = {'ProcessedFile': open(output_filename, 'rb')}
    values = {
        'QueueId': queue_id,
        'FileType':file_type,
        }
    print(values)
    req_url = f'{config.API_SERVER_IP}{config.API_PROCESSED_PATH}'
    try:
        res = requests.post(
            url=req_url, files=files, data=values, headers={"X-App-Credential": f"{config.API_KEY}"})
        if res.status_code == 401:
            print(f'{res.status_code} Invalid Credential. Please Fix `API_KEY`. Waiting 20s')
            time.sleep(20)
            return False
        res = json.loads(res.content)
        print(f"send_processed_image response: {res}")
        if(res['Status']=='success'):
            return True
        elif(res['Status']=='fail'):
            return False
        else:
            return True
    except Exception as e:
        print(e)
        return False

if __name__ == "__main__":
    
    while True:
        
        req_url = f'{config.API_SERVER_IP}{config.API_UNPROCESSED_PATH}'
        try:
            res = requests.get(url=req_url)
            if res.status_code == 401:
                print(f'{res.status_code} Invalid Credential. Please Fix `API_KEY`. Waiting 60s')
                time.sleep(60)
                continue
            res = json.loads(res.content)
            print(res)
        except Exception as err:
            print(
                f"Can't Connect to InfiniteLoop API server at {req_url}...Waiting 100s")
            time.sleep(100)
            continue
        try:
            file_type= res['FileType']
        except:
            print("FileType not found. Default as zip")
            file_type='zip'
        try:
            blur_level=int(res['BlurStrength'])
        except:
            print("BlurStrength not found. Default as 43")
            blur_level=43
        #added by Jahirul
        try:
            dilation_iteration=int(res['Dilation_iteration'])
        except:
            print("Dilation iteration is not found. Default as 12")
            dilation_iteration=1
        #################
        try:
            output_style=res['OutputStyle']
            if(output_style=='blurred'):
                output_type='blur_path'
            elif (output_style=='transparent'):
                output_type='trasparent_path'
            elif (output_style=='white'):
                output_type='white_path'
            else:
                output_type='blur_path'
        except:
            print("OutputStyle not found. Default as blur")
            output_type='blur_path'
        try:
            output_size=res['OutputSize']
        except:
            print("OutputSize not found. Default as original")
            output_size='original'
        processed_path=run_process(output_type,file_type,blur_level,output_size,dilation_iteration)
        '''
        req_url = f'{config.API_SERVER_IP}{config.API_UNPROCESSED_PATH}'
        try:
            res = requests.get(url=req_url, headers={"X-App-Credential": f"{config.API_KEY}"})
            if res.status_code == 401:
                print(f'{res.status_code} Invalid Credential. Please Fix `API_KEY`. Waiting 60s')
                time.sleep(60)
                continue
            res = json.loads(res.content)
            print(res)
        except Exception as err:
            print(
                f"Can't Connect to InfiniteLoop API server at {req_url}...Waiting 100s")
            time.sleep(100)
            continue
        try:
            res['Status']
        except Exception as err:
            print("Status is not found")
            continue
        if(res['Status'] == 'empty'):
            print('No image to process')
            if(res['WaitTime'] is not None):
                print(f"waiting {res['WaitTime']} seconds")
                time.sleep(res['WaitTime'])
                continue
            print(f'waiting 60 seconds')
            time.sleep(60)
        elif(res['Status'] == 'invalid credential'):
            print(f"{res['Status']}. Please Fix API_KEY")
            time.sleep(60)
        else:
            try:
                queue_id = res['QueueId']
                file_url = res['Url']
            except:
                print("Url or queue_id not found. Skipping Process")
                continue
            try:
                file_type= res['FileType']
            except:
                print("FileType not found. Default as zip")
                file_type='zip'
            try:
                blur_level=int(res['BlurStrength'])
            except:
                print("BlurStrength not found. Default as 43")
                blur_level=43
            #added by Jahirul
            try:
                dilation_iteration=int(res['Dilation_iteration'])
            except:
                print("Dilation iteration is not found. Default as 12")
                dilation_iteration=12
            #################
            try:
                output_style=res['OutputStyle']
                if(output_style=='blurred'):
                    output_type='blur_path'
                elif (output_style=='transparent'):
                    output_type='trasparent_path'
                elif (output_style=='white'):
                    output_type='white_path'
                else:
                    output_type='blur_path'
            except:
                print("OutputStyle not found. Default as blur")
                output_type='blur_path'
            try:
                output_size=res['OutputSize']
            except:
                print("OutputSize not found. Default as original")
                output_size='original'
            
            try:
                add_time = res['AddTime']
            except Exception as err:
                print("AddTime not found")
            file_url_ext=file_url.split('.')[-1]
            print(f"File_Url:{file_url}")
            print(f"File_Url_ext:{file_url_ext}")
            print(f"File_Type:{file_type}")
            if(file_type!=file_url_ext):
                print("file_type does not match file_url extention. Dropping process")
                continue
            if file_type=="zip":
                filename=f"{data_path}/inputdata.{file_type}"
            else :
                filename=f"{data_path}/inputdata/{queue_id}.{file_type}"
            print(filename)
            try:
                urllib_req.urlretrieve(file_url, filename )
            except Exception as err:
                print("Unable to download image from CDN,droping process")
                continue

            processed_path=run_process(output_type,file_type,blur_level,output_size,dilation_iteration)
            if file_type!='zip' and output_type=='trasparent_path':
                file_type='png'
            if not send_processed_file(processed_path,queue_id,file_type):
                # retry again if sending file fails
                print(
                    f"Failed to send post request to {config.API_SERVER_IP} . Trying again after 10 second")
                time.sleep(10)
                if not send_processed_file(processed_path,queue_id,file_type):
                     print(f"Unable to send Processed File to {config.API_SERVER_IP}. Dropping process")'''