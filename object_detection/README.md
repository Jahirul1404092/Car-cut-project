# Crop Car

1. Get the weight file form following link [yolov5n6](https://chowagiken.sharepoint.com/:u:/s/proj_car_defect_detection_/EQwmfqyKKGdPpyDwMrFiLLQBnrZL8yWuY87inAp1XpmkiA?e=gJ8bBp). And place it to the current directory.
2. Next

```bash
$ pip install -r requirements.txt
``` 
3. In the following command, give the input path `--source` and output path `--crop_output`. 

```python
python detect.py --weights "yolov5n6.pt" 
                 --source "../data/inputdata"        # image directory
                 --crop_output "../data/pre_process"   # a folder name crop will be created and the cropped images will be saved there
                 --save-crop 
                 --only-max-area 
                 --classes 2 5 7
```

