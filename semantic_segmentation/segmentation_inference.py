'''
Author: M.Innat, Developer @Chowagiken
Mail: mohammed.innat@chowagiken.co.jp
Web: https://github.com/innat
'''
# general imports 
import os, cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from time import time
from colorama import Fore
from os.path import basename, normpath
from skimage.transform import resize
import sys

# machine learning libs
import tensorflow as tf
print(f"{Fore.YELLOW}_"*50)
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import segmentation_models as sm
import matplotlib.pyplot as plt

# If no GPU is found, physical_devices would be None.
physical_devices = tf.config.list_physical_devices('GPU') # it's only meant to work with GPU.
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: 
    pass

# Semantic Segmentation 'model.h5' should be located in the current directory 
MODEL_WEIGHT_PATH = 'model.h5'

# Input files should be located in the following directory 
DIRECTORY = '../data/semantics/input'
DIRECTORY2 = '../data/inputdata'
DIRECTORY3 = '../data/semantics/bigger_mask'
EXT = ('.jpg', '.png', '.jpeg')

# Simple Class with required Functions for Inferences
class BinarySegmentation(object):
    def __init__(self, model_weight_path):
        # Current task only have one target 
        self.CLASSES = ['car']
        
        # we use 1024 while training the semantic seg. model. 
        # bigger input, better resutls. 
        # use much bigger for better output. 
        self.TargetHeight = 1024
        self.TargetWidth  = 1024
        
        # define network parameters
        self.BACKBONE = 'efficientnetb0' 
        
        self.n_classes = len(self.CLASSES)
        self.activation = 'sigmoid'
        self.model = self.get_model(weight_path=model_weight_path)

    def get_model(self, weight_path):
        # create model
        model = sm.Unet(
            self.BACKBONE,
            input_shape=(self.TargetHeight, self.TargetWidth, 3),
            classes=self.n_classes,
            encoder_weights=None,
            decoder_block_type='upsampling',
            activation=self.activation
        )
        
        model.load_weights(weight_path)
        return model
    
    @staticmethod
    def read_image(path):
        return cv2.imread(path)
    
    @staticmethod
    def pre_process(sample, height, width):
        sample = cv2.resize(sample, (height, width))
        sample = sample / 255.
        sample = sample[None, ...]
        return sample
    
    def post_process_white_bg(self, y_pred, mask_image):
        y_pred = cv2.resize(y_pred[0].astype('float32'), (w, h))
        y_pred = cv2.cvtColor(y_pred, cv2.COLOR_GRAY2RGB) 
        y_pred = y_pred.astype(np.uint8)*255
        
        # we used bigger mask from instance segmentation model
        # and lower-cropped to the output of 
        # semantic segmentation output. In that way, there will 
        # be no overlap mask of the semantic seg. output. 
        # For better understanding, please contanct with the script writer. 
        y_pred = y_pred *- mask_image if cv2.countNonZero(mask_image[:,:,0]) else y_pred
        y_pred = self.remove_minor_maks(y_pred)
        return y_pred
    
    @staticmethod
    def post_process_transparent_bg(y_pred):
        mask = cv2.cvtColor(y_pred, cv2.COLOR_RGB2GRAY) 
        mask = mask.astype(np.uint8)
        return mask
    
    @staticmethod
    def post_process_blurry_bg(y_pred, orig_img, blur_level):
        y_pred = y_pred.astype(np.uint8)*255
        blurred_img = cv2.GaussianBlur(orig_img, (blur_level, blur_level), 0)
        return y_pred, blurred_img

    def inference(self, image_path, image_path2, image_path3, blur_level):
        # Get File 
        img = self.read_image(image_path)[:,:,::-1]
        orig_img = self.read_image(image_path2)[:,:,::-1]
        mask_img = self.read_image(image_path3)[:,:,::-1]
        global h,w,c
        h, w, c = img.shape
        
        # Do Preprocess 
        # and Inference
        process_sample = self.pre_process(img, self.TargetHeight, self.TargetWidth)
        y_pred = (self.model.predict(process_sample) > 0.5).astype(np.int32)

        # It is possible that Mask-RCNN couldn't detect anything on sample X. 
        # or
        # It is possible that UNet able to detect pattern on the same sample X.
        if (
            cv2.countNonZero(cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)) or 
            cv2.countNonZero(y_pred[0])
        ): 
            # Do Postprocess 
            # target object with white background
            y_pred = self.post_process_white_bg(y_pred, mask_img)
            white_bg = np.where(y_pred, img, 255) 
            
            # Do Postprocess 
            # target object with transparent background
            mask = self.post_process_transparent_bg(y_pred)
            trans_bg = np.dstack(( white_bg, mask ))

            # Do Postprocess 
            # target object with blurry background
            y_pred, blurred_img = self.post_process_blurry_bg(y_pred, orig_img, blur_level)
            blurred_bg = np.where(y_pred, orig_img, blurred_img)  
        else:
            white_bg = img
            blurred_bg = img
            trans_bg = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA) 
            
        return trans_bg, white_bg, blurred_bg
    
    def remove_minor_maks(self, mask_rgb):
        '''
        input: mask: it should be shape of (h, w, 3) 
        return mask: it should be shape of (h, w, 3)
        what this function does? 
            ans. this function will compute each mask area and only return 
            the bigger area or bigger mask. But if there is only one mask, 
            then input and return should be same.

        Ref. https://stackoverflow.com/a/58070769/9215780
        Adopted and Modified: M.Innat
        '''

        mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
        mask_gray = mask_gray.astype("uint8")
        thresh    = cv2.threshold(mask_gray, 0,255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        cnts      = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts      = cnts[0] if len(cnts) == 2 else cnts[1]

        total_masks      = []
        total_mask_areas = []
        
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            mask       = np.zeros(mask_rgb.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [c], [255,255,255])
            mask       = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            pixels     = cv2.countNonZero(mask)
            total_mask_areas.append(pixels)
            total_masks.append(mask)
             
        if len(total_mask_areas) > 1:
            max_pixel_index = total_mask_areas.index(max(total_mask_areas))
            max_pixel_area  = total_masks[max_pixel_index]
            
            kernel = np.ones((10,10),np.uint8)
            # max_pixel_area = cv2.erode(max_pixel_area, kernel,iterations = 1)
            # max_pixel_area = cv2.dilate(max_pixel_area, kernel,iterations = 12) #standare 12
            max_pixel_area = cv2.dilate(max_pixel_area, kernel,iterations = dilation_iteration) #standare 12
            
            max_pixel_area = cv2.cvtColor(max_pixel_area, cv2.COLOR_GRAY2RGB) 
            return max_pixel_area
        else:
            '''Closing is reverse of Opening, Dilation followed by Erosion. 
            It is useful in closing small holes inside the foreground objects, 
            or small black points on the object.
            '''
            # ref. https://stackoverflow.com/a/10317883
            # ref. https://opencv24-python-tutorials.readthedocs.io/en/latest/index.html
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
            kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
            mask_rgb = cv2.morphologyEx(mask_rgb, cv2.MORPH_CLOSE, kernel)
            
            kernel = np.ones((10,10),np.uint8)
            # mask_rgb = cv2.erode(mask_rgb, kernel,iterations = 1)
            # mask_rgb = cv2.dilate(mask_rgb, kernel,iterations = 12) #standard 12
            mask_rgb = cv2.dilate(mask_rgb, kernel,iterations = dilation_iteration) #standard 12
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_GRAY2RGB)
            return mask_rgb


def main(
    MODEL_WEIGHT_PATH='model.h5',
    DIRECTORY = '../data/semantics/input',
    DIRECTORY2 = '../data/inputdata',
    DIRECTORY3 = '../data/semantics/bigger_mask',
    DIRECTORY_TRANSPARENT_OUTPUT = '../data/output/transparent_bg',
    DIRECTORY_WHITE_BG_OUTPUT = '../data/output/white_bg',
    DIRECTORY_BLUR_BG_OUTPUT = '../data/output/blur_bg',
    BLUR_LEVEL=43,
    OUTPUT_SIZE='original',
    DILATION_ITERATION=12
    ):

    global dilation_iteration
    dilation_iteration=DILATION_ITERATION
    # initialize the binary semantic segmentation model.
    binary_seg = BinarySegmentation(MODEL_WEIGHT_PATH)
    paths = sorted(
        [
            os.path.join(dirpath,filename) 
            for dirpath, _, filenames in os.walk(DIRECTORY) 
            for filename in filenames if filename.endswith(EXT) and not filename.startswith(".")
        ],
        key = lambda i: os.path.splitext(os.path.basename(i))[0]
    )
    new_paths = []
    for each_a in paths:
        if not '.ipynb_checkpoints' in each_a.split('/'):
            new_paths.append(each_a)
    
    new_paths = sorted(new_paths, key = lambda i: os.path.splitext(os.path.basename(i))[0])
    paths = new_paths

    paths2 = sorted(
        [
            os.path.join(dirpath,filename) 
            for dirpath, _, filenames in os.walk(DIRECTORY2) 
            for filename in filenames if filename.endswith(EXT) and not filename.startswith(".")
        ],
        key = lambda i: os.path.splitext(os.path.basename(i))[0]
    )
    new_paths = []
    for each_a in paths2:
        if not '.ipynb_checkpoints' in each_a.split('/'):
            new_paths.append(each_a)
    
    new_paths = sorted(new_paths, key = lambda i: os.path.splitext(os.path.basename(i))[0])
    paths2 = new_paths

    paths3 = sorted(
        [
            os.path.join(dirpath,filename) 
            for dirpath, _, filenames in os.walk(DIRECTORY3) 
            for filename in filenames if filename.endswith(EXT) and not filename.startswith(".")
        ],
        key = lambda i: os.path.splitext(os.path.basename(i))[0]
    )
    new_paths = []
    for each_a in paths3:
        if not '.ipynb_checkpoints' in each_a.split('/'):
            new_paths.append(each_a)
    
    new_paths = sorted(new_paths, key = lambda i: os.path.splitext(os.path.basename(i))[0])
    paths3 = new_paths

    print(f"{Fore.BLUE}_"*50)
    print("[Program Running: 3rd and Final Stage].")
    print("Semantic Segmentation Inferencing (Model: UNet [EfficientNet] )")
    
    # for computing the execution time.
    start_time = time()
    
    for i, (path, path2, path3) in enumerate(
        tqdm(zip(paths, paths2, paths3), total=len(paths2))
    ):
        
        # running inference.
        trans_bg, white_bg, blurred_bg = binary_seg.inference(path, path2, path3, BLUR_LEVEL)
        
        # appropriate folder create for saving the inference results.
        temp_dir = '/'.join(
            os.path.dirname(
                os.path.normpath(path)
            ).split(os.sep)[3:] # exclude ./data/semantics/input/ part from path.
        )
        
        trans_out = os.path.join(DIRECTORY_TRANSPARENT_OUTPUT, temp_dir)
        os.makedirs(trans_out, exist_ok=True)
        
        white_out = os.path.join(DIRECTORY_WHITE_BG_OUTPUT, temp_dir)
        os.makedirs(white_out, exist_ok=True)
        
        blurr_out = os.path.join(DIRECTORY_BLUR_BG_OUTPUT, temp_dir)
        os.makedirs(blurr_out, exist_ok=True)
        
        # filename (i.e. sample.jpg)
        file_name = basename(normpath(path))
        trans_file_name = file_name.split('.')[0] + '.png'
        
        # resizing the output image.
        if OUTPUT_SIZE != "original":
            # OUTPUT_SIZE: (4000x2000) or (4000X2000)
            dim_h, dim_w = map(int, OUTPUT_SIZE.replace('X', 'x').split('x'))
            
            trans_bg = cv2.resize(trans_bg, (dim_h, dim_w))
            white_bg = cv2.resize(white_bg, (dim_h, dim_w))
            blurred_bg = cv2.resize(blurred_bg, (dim_h, dim_w))
        
        cv2.imwrite(os.path.join(trans_out, trans_file_name), trans_bg[:,:,[2,1,0,3]])
        cv2.imwrite(os.path.join(white_out, file_name), white_bg[:,:,::-1])
        cv2.imwrite(os.path.join(blurr_out, file_name), blurred_bg[:,:,::-1])

    total_time = time() - start_time
    print("Total {} images takes {:.3f} seconds".format(len(paths), total_time))
    print("Per image takes avg {:.3f} seconds".format(total_time/len(paths)))
    print(f"{Fore.BLACK}_"*50)
    print()

if __name__ == "__main__":
    main()
