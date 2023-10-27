gsutil -m  cp  gs://ifl_car/models/binary_segment/model.h5 semantic_segmentation/.
gsutil -m  cp  gs://ifl_car/models/yolov5/yolov5n6.pt object_detection/.
gsutil -m  cp  gs://ifl_car/models/instance_segment/model_final.pth instance_segmentation/.
gsutil -m  cp  gs://ifl_car/packages/torch-1.9.0+cu111-cp37-cp37m-linux_x86_64.whl ./
gsutil -m  cp  gs://ifl_car/packages/torchvision-0.10.0+cu111-cp37-cp37m-linux_x86_64.whl ./
gsutil -m  cp  gs://ifl_car/packages/detectron2-0.6+cu111-cp37-cp37m-linux_x86_64.whl ./

# gsutil -m  cp  gs://car_cut/models/binary_segment/model.h5 semantic_segmentation/.
# gsutil -m  cp  gs://car_cut/models/yolov5/yolov5n6.pt object_detection/.
# gsutil -m  cp  gs://car_cut/models/instance_segment/model_final.pth instance_segmentation/.
# gsutil -m  cp  gs://car_cut/packages/torch-1.9.0+cu111-cp37-cp37m-linux_x86_64.whl ./
# gsutil -m  cp  gs://car_cut/packages/torchvision-0.10.0+cu111-cp37-cp37m-linux_x86_64.whl ./
# gsutil -m  cp  gs://car_cut/packages/detectron2-0.6+cu111-cp37-cp37m-linux_x86_64.whl ./
