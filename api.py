from flask import Flask,send_file
from flask_restful import Resource, Api, reqparse
import werkzeug
# import time
from main import run_car_cut
import urllib.request as urllib_req
import os


app = Flask(__name__)
api = Api(app)

class ServerStatus(Resource):
    def get(self):
        return {'status': 'active', 'msg':'car cut segmentation Api is active'}

class PredictImageUrl(Resource):
    def post(self):
        img_path="./data/inputdata/uploaded.jpeg"
        parser = reqparse.RequestParser()
        parser.add_argument(
            'Url',
            type=str,
            required=True,help='Url for image to car_cut,Url is required')
        args = parser.parse_args()
        urllib_req.urlretrieve(args['Url'],img_path)
        print("Image URL Received")
        # infer=run_car_cut()
        # prediction= infer.single_predict(img_path)
        output=run_car_cut()
        return_file=f"{output['blur_path']}/uploaded.png"
        # print(return_file)
        return_file=send_file(return_file, mimetype='image/png')
        # print(type(return_file))
        return return_file
        
# class PredictImageFile(Resource):
#     def post(self):
#         img_path="uploaded.jpeg"
#         parser = reqparse.RequestParser()
#         parser.add_argument(
#             'File',
#             type=werkzeug.datastructures.FileStorage,
#             location='files',required=True,help='image file is required')
#         args = parser.parse_args()
#         image_file = args['File']
#         image_file.save(img_path)
#         print("Image File Received")
#         # infer=Inferance()
#         run_car_cut()
#         # prediction= infer.single_predict(img_path)
#         return {
#             'status': 'success',
#             'msg':'image carCut finished',
#             # 'result':prediction
#             }

api.add_resource(ServerStatus, '/api/status/')
api.add_resource(PredictImageUrl, '/api/carcut/url/')
# api.add_resource(PredictImageFile, '/carcut/file/')


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT", 5000)),debug=True)
