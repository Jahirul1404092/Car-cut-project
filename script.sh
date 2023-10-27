# cd object_detection
# python -W ignore detect.py --weights "yolov5n6.pt" --source "../data/inputdata" --crop_output "../data/instance/input"  --save-crop --only-max-area --classes 2 5 7

# cd ..
cd instance_segmentation
python -W ignore instance_inference.py

cd ..
cd semantic_segmentation
python -W ignore segmentation_inference.py

# cd ..
# cd data/
# rm -r instance
# rm -r semantics
