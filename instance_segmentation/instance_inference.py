'''
Author: Instance model trained : Munna-san, Ex-Developer @Chowagiken
Modified and Adopted: M.Innat, Developer @Chowagiken (https://github.com/innat)
Mail: mohammed.innat@chowagiken.co.jp
'''

# general imports 
import os, cv2
import numpy as np 
from glob import glob
from tqdm import tqdm
from time import time
from colorama import Fore
from os.path import basename, normpath
from scipy.ndimage.filters import gaussian_filter

# machine learning libs
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Instance Segmentation 'model_final.pth' should be located in the current directory 
MODEL_WEIGHT_PATH = 'model_final.pth'

# Input files should be located in the following directory 
INPUT_DIRECTORY   = '../data/inputdata' #'../data/instance/input'
EXT = ('.jpg', '.png', '.jpeg')


# Simple Class with required Functions for Inferences
class InstanceSegmentation(object):
    def __init__(self, model_weight_path,device):
        cfg = get_cfg()
        # number of our target class only 1, i.e. front car
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        
        # to run on CPU device, in detectron we do as follows: 
        if device=='cpu':
            cfg.MODEL.DEVICE='cpu'
        
        # the final instance segmentation model is Mask-RCNN 101 
        yaml_config = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(yaml_config))
        cfg.MODEL.WEIGHTS = model_weight_path
        self.predictor = DefaultPredictor(cfg)
        
    def remove_inner_masks_from_target_mask(self, large_mask, smaller_masks):
        '''Sometimes, the model can detect object through other object. 
        For example, in our car cut program, the front car is needed to cut 
        or segment only. Now, the instance segmentation model can detect car(s) 
        that are visible trough the target car's window. In that case, we like 
        to skip that detection. And this function just do that task. In our case, 
        the target mask is always the bigger mask and anything else are smaller 
        than that. 
        
        Note: in future, it may require to remove everthing that are visible from 
        target car's windows. In that case, we may need new method/model to do that. 
        '''
        return large_mask, large_mask + smaller_masks
    

    def get_largest_and_smallest_obj_index(self, outputs):
        '''It simply process the output of instance segmentaiton model from detectron2 
        library. Its task is to compute the bigger mask and smaller masks. Note, the 
        bigger mask should be only one mask and rest of all are smaller masks.
        '''
        Box= outputs["instances"].pred_boxes
        area = Box.area().to('cpu').numpy()
        large_box_index = np.argmax(area)
        smaller_indexes = np.where(area != area[large_box_index])[0]
        return large_box_index, smaller_indexes
    
    def get_smallers_car_mask(self, outputs, smaller_indexes):
        '''It simply process the output of instance segmentaiton model from detectron2 
        library. Its task is to use ouotput of instance segmentation model and smaller 
        mask index (from get_largest_and_smallest_obj_index function above) and return 
        smaller mask.
        
        As instance segmentation, it gives smaller masks with shape 
        `(..., h, w, num_of_mask)`. This function simply collapse all channel into one.
        And create a 2d binary (0:background, 255:smaller masks) mask. 
        '''
        masks = outputs['instances'].pred_masks[smaller_indexes].to('cpu').numpy()
        # add masks
        h, w = masks.shape[1:]
        # add masks
        output_mask = np.zeros((h, w))
        for mask in masks:
            output_mask += mask
        output_mask *=255
        output_mask = output_mask.clip(0, 255).astype("uint8")
        return output_mask
    
    def soft_blur_with_mask(self, im, mask, blur_level):
        '''It's one of the preprocess on the output of instance segmentation model. 
        Its task is to blur the every masks expect the bigger mask, which is in our 
        case is the target mask. 
        '''
        blurred_im = cv2.GaussianBlur(im, (blur_level,blur_level), 0)
        mask = np.stack((mask,)*3, axis=-1)
        image = np.where(mask == 255, blurred_im, im)
        return image
    
    def colorize_bg_cars(self, im,mask):
        '''Similar to soft_blur_with_mask function, instead of blurring, it set solid colors. 
        '''
        result = im.copy()
        result[mask==255] = [0,255,255]
        return result
    
    def inference(self, image_path, blur_level):
        # read and pass to predictors. 
        img = cv2.imread(image_path)
        outputs = self.predictor(img)
        
        # (a quick fix to handle no boject detection cases)
        # (todo [innat]: find better and elegant approach to address.)
        if len(outputs['instances']):
            # get large and smaller maks index and correspond mask.
            largest_car_index, smaller_car_indexs = self.get_largest_and_smallest_obj_index(outputs)
            largest_car_mask  = outputs['instances'].pred_masks[largest_car_index].to('cpu').numpy()
            smaller_car_masks = outputs['instances'].pred_masks[smaller_car_indexs].to('cpu').numpy()
            smaller_car_masks = smaller_car_masks.transpose(1,2,0)
            
            # If more than one mask detected. 
            # If smaller mask exist. 
            if smaller_car_masks.shape[-1]: 
                # (Issue 1)
                # Option 1
                # largest_car_mask, smaller_car_masks = self.remove_inner_maks_from_target_mask(largest_car_mask,
                #                                                                               smaller_car_masks)
                # smaller_car_masks = smaller_car_masks.argmax(axis=-1)
                # smaller_car_masks = np.where(smaller_car_masks == 0, 0, 255).clip(0, 255).astype("uint8")

                # Fix issue 1
                # Option 2
                smaller_car_masks2d = np.zeros_like(smaller_car_masks[:,:,0]).astype(np.uint8)
                for i in range(smaller_car_masks.shape[-1]):
                    array = smaller_car_masks[:,:, i]
                    array = array.astype(np.float32)

                    # dilation becomes important.
                    # it reduce the gap among the smaller masks. So that, after blurring or 
                    # setting solid car, no car part would be visible. 
                    array = cv2.dilate(
                        array, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
                    ) # 11, 13, 15
                    array = array.astype(bool)
                    smaller_car_masks2d += array

                smaller_car_masks = smaller_car_masks2d
                smaller_car_masks = np.where(smaller_car_masks == 0, 0, 255).clip(0, 255).astype("uint8")
                largest_car_mask, smaller_car_masks = self.remove_inner_masks_from_target_mask(
                    largest_car_mask,
                    smaller_car_masks
                )

                blur_cars = self.soft_blur_with_mask(img[:, :, ::-1], smaller_car_masks, blur_level)
                # colr_cars = self.colorize_bg_cars(img[:, :, ::-1], smaller_car_masks)

                '''We did apply dilation on the bigger mask too. 
                Because, we plan to use it in next stage where we use semantic segmentation.
                '''
                largest_car_mask = largest_car_mask.astype(np.float32)
                # ref https://stackoverflow.com/a/60496941/9215780
                paddedMask = cv2.dilate(largest_car_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100))) 
                paddedMask = cv2.cvtColor(paddedMask, cv2.COLOR_GRAY2RGB)*255
                return blur_cars, paddedMask
            else:
                # If no smaller masks detected or there is only one bigger mask.
                # Then we don't need to do any sort of post processing on the output of instance segmentation. 
                # The semantic segmentation is well generalized. 
                # The semantic segmentation model is trained on mostly binary segmentation type dataset. 
                return img[:, :, ::-1], np.zeros_like(img[:,:,0]).astype(np.uint8)
        else:
            return img[:, :, ::-1], np.zeros_like(img[:,:,0]).astype(np.uint8)
        
 

        
def main(
    MODEL_WEIGHT_PATH="../instance_segmentation/model_final.pth",
    INPUT_DIRECTORY   = '../data/inputdata', #'../data/instance/input'
    OUTPUT_DIRECTORY  = '../data/semantics/input',
    OUTPUT_DIRECTORY2 = '../data/semantics/bigger_mask',
    DEVICE= '', # keep empty to run on GPU, use 'cpu' to run on cpu
    BLUR_LEVEL=43
):
    # initialize the instance segmentation model.
    instance_seg = InstanceSegmentation(MODEL_WEIGHT_PATH,DEVICE)
    paths = sorted(
        [
            os.path.join(dirpath,filename) 
            for dirpath, _, filenames in os.walk(INPUT_DIRECTORY) 
            for filename in filenames if filename.endswith(EXT) and not filename.startswith(".")
        ],
        key = lambda i: os.path.splitext(os.path.basename(i))[0]
    )
    
    print(f"{Fore.BLUE}_"*50)
    print("[Program Running: 2nd Stage] Instance Segmentation Inferencing.")
    print("Removing Background Cars (Model: Mask-RCNN-101)")

    # for computing the execution time.
    # [TODO:] find better approach if available.
    start_time = time()

    for path in tqdm(paths):
        # running inference.
        infer_result, paddedMask = instance_seg.inference(path, blur_level=BLUR_LEVEL)
        
        # appropriate folder create for saving the inference results.
        temp_dir = '/'.join(
            os.path.dirname(
                os.path.normpath(path)
            ).split(os.sep)[2:] # exclude ./data/inputdata/ part from path.
        )
        parent_path = os.path.join(OUTPUT_DIRECTORY, temp_dir)
        os.makedirs(parent_path, exist_ok=True)
        
        parent_path2 = os.path.join(OUTPUT_DIRECTORY2, temp_dir)
        os.makedirs(parent_path2, exist_ok=True)
 
        # filename (i.e. sample.jpg)
        file_name = basename(normpath(path))
        
        # savings.
        cv2.imwrite(os.path.join(parent_path, file_name), infer_result[:,:,::-1])
        cv2.imwrite(os.path.join(parent_path2, file_name), paddedMask)

    total_time = time() - start_time
    print("Total {} images takes {:.3f} seconds".format(len(paths), total_time))
    print("Per image takes avg {:.3f} seconds".format(total_time/len(paths)))
    print()

if __name__ == "__main__":
    main()
