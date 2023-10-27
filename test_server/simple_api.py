#using carCut Conda env
from datetime import datetime
import os
import uuid
from flask import Flask, render_template, session, request,send_file
from werkzeug.utils import secure_filename
from pathlib import Path
from google.cloud import storage
from firebase import firebase

app = Flask(__name__)


# @app.route('/api/ai/exterior/', methods=['GET'])
# def UnprocessedZip():
#     return {'Status': 'empty','WaitTime':2}

@app.route('/')
def ProcessedFile():
    current_dict={
            'FileType':'zip',
            'OutputStyle':'blur_path',
            'BlurStrength':'43',
            'OutputSize' : 'original',
            'Dilation_iteration':'-1'
        }
    return current_dict

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)