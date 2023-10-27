# Car Segmentation 

This program is developed to segment a single car from an image. Image may contain multiple cars. So, this program will follow a series of deep neural networks as a complete solutions. Here are the steps:

- Stage 1: Detect and Crop the target car.
- Stage 2: Ensure the target car is only appearing in the image.
- Stage 3: Segment the target car and place to while backgrouond/transparent background. 

This program will do these series of solutions simultaneously. Each of the stage uses different type of Deep Learning methods. 

## Directory Structure

    ├── object_detection          # Crop `data>inputdata` images using YoloV5 
    ├── data
    │   ├── inputdata             # Input Original Images
    │   ├── instance              # Output of Cropped Image from Object Det. Modle
    │   ├── semantics             # Output of Instance Segmentaiton Model 
    │   ├── output      
    │   │   ├── transparent_bg    # Output Transparent Background Images
    │   │   └── white_bg          # Output White Background Images
    ├── semantic_segmentation     # Semantic Segmentation Models 
    ├── instance_segmentation     # Instance Segmentation Models 
    ├── requirements.txt          # Required Libraries
    ├── README.md                 # Readme with Execusion Instructions
    ├── prepare.sh                # Bash Script to download required models and place those in the proper directory
    └── script.sh                 # Bash Script to do the automatic cropping and segmenting


## Setup

You must have GPU as we're going to use a series of deep networks. So, update your GPU with [latest GPU drivers](https://www.nvidia.com/download/index.aspx?lang=en-us). Also, some other utility to enable your GPU to interact with Deep Learning libraries. This utilities are as follows and they must have the following version.

- CUDA® Toolkit: 11.2
- cuDNN SDK 8.1.0

1. Download linux version of [Anaconda](https://docs.anaconda.com/anaconda/install/) and install it in your linux system, [installation process](https://www.youtube.com/watch?v=5mDYijMfSzs). After successfull installation, next create a new enivironment with python 3.7. 

```
conda create -n my_env python=3.7
```

After that, activate it as follows
```
conda activate my_env
```

2. Currently the Model weight files and some essential files are stored in GCP. To get them, run the following command. 
```bash
bash prepare.sh
```

3. Next, run the following command 
```bash
pip install -r requirements.txt
```

In summary, after installing anaconda into your linux, open up your terminal and execute the above commands as follows. 

![carbon](https://user-images.githubusercontent.com/81627860/148677799-a87815b3-44d8-48e8-bce8-7a818441be22.png)


## Program Execution 

1. Put input image in `data>inputdata` directory
2. Run command 
```bash
bash script.sh
``` 
The execution should look like as follows: 

```python
jupyter@mlinnat:~/proj_car_segmentation$ bash script.sh 
Sofware: Car Cut - Removing Background of Car.
Developed by CHOWA GIKEN Corporation (https://www.chowagiken.co.jp/)
  _____ _                              _ _
 / ____| |                            (_) |
| |    | |__   _____      ____ _  __ _ _| | _____ _ __
| |    |  _ \ / _ \ \ /\ / / _` |/ _` | | |/ / _ \  _ \ 
| |____| | | | (_) \ V  V / (_| | (_| | |   <  __/ | | |
 \_____|_| |_|\___/ \_/\_/ \__,_|\__, |_|_|\_\___|_| |_|
                                  __/ |                 
                                 |___/                  
_____________________________________
[Program Running: 1st Stage] Single Car Object Detection (Model: Yolo-V5)
Total Image Found:  3
Total Inference Time:  0.0462 second

__________________________________________________
[Program Running: 2nd Stage] Instance Segmentation Inferencing - Removing Background Cars (Model: Mask-RCNN-101)
100%|█████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.10it/s]
Total 3 images takes 2.731 seconds
Per image takes avg 0.910 seconds

__________________________________________________
Segmentation Models: using `tf.keras` framework.
[Program Running: 3rd and Final Stage] Semantic Segmentation Inferencing (Model: UNet [EfficientNet] )
100%|█████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:07<00:00,  2.36s/it]
Total 3 images takes 7.072 seconds
Per image takes avg 2.357 seconds
```

3. It will generates outputs and saves them in the `data>output>transparent_bg` and `data>output>white_bg` directories


