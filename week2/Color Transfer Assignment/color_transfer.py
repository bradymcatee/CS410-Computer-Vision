import cv2
import numpy as np
import sys
import math

def convert_color_space_BGR_to_RGB(img_BGR):
    if img_BGR is None:
        raise ValueError("Image not found or unable to load")

    return img_BGR[:,:,::-1]

def convert_color_space_RGB_to_BGR(img_RGB):
    if img_RGB is None:
        raise ValueError("Image not found or unable to load")

    return img_RGB[:,:,::-1]

def convert_color_space_RGB_to_Lab(img_RGB):
    '''
    convert image color space RGB to Lab
    '''
    rgb_to_lms_matrix = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]
    ], dtype=np.float32)
    
    img_RGB = img_RGB / 255.0
    img_LMS = np.dot(img_RGB.reshape(-1, 3), rgb_to_lms_matrix.T)
    img_LMS = np.log(np.maximum(img_LMS, 1e-6)) 
    
    lms_to_lab_matrix = np.array([
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
        [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
        [1/np.sqrt(2), -1/np.sqrt(2), 0]
    ], dtype=np.float32)
    
    img_Lab = np.dot(img_LMS, lms_to_lab_matrix.T).reshape(img_RGB.shape)
    return img_Lab

def convert_color_space_Lab_to_RGB(img_Lab):
    '''
    convert image color space Lab to RGB
    '''
    lab_to_lms_matrix = np.array([
        [1/np.sqrt(3), 1/np.sqrt(6), 1/np.sqrt(2)],
        [1/np.sqrt(3), 1/np.sqrt(6), -1/np.sqrt(2)],
        [1/np.sqrt(3), -2/np.sqrt(6), 0]
    ], dtype=np.float32)
    
    img_LMS = np.dot(img_Lab.reshape(-1, 3), lab_to_lms_matrix.T)
    img_LMS = np.exp(img_LMS)
    
    lms_to_rgb_matrix = np.array([
        [4.4679, -3.5873, 0.1193],
        [-1.2186, 2.3809, -0.1624],
        [0.0497, -0.2439, 1.2045]
    ], dtype=np.float32)
    
    img_RGB = np.dot(img_LMS, lms_to_rgb_matrix.T).reshape(img_Lab.shape)
    img_RGB = np.clip(img_RGB * 255.0, 0, 255).astype(np.uint8)
    return img_RGB



def color_transfer(img_RGB_source, img_RGB_target):
    print('===== color_transfer_in_Lab =====')
    img_Lab_source = convert_color_space_RGB_to_Lab(img_RGB_source)
    img_Lab_target = convert_color_space_RGB_to_Lab(img_RGB_target)
    
    mean_source = np.mean(img_Lab_source, axis=(0, 1))
    std_source = np.std(img_Lab_source, axis=(0, 1))
    mean_target = np.mean(img_Lab_target, axis=(0, 1))
    std_target = np.std(img_Lab_target, axis=(0, 1))
    
    img_Lab_result = (img_Lab_source - mean_source) * (std_target / std_source) + mean_target
    
    img_RGB_result = convert_color_space_Lab_to_RGB(img_Lab_result)
    return img_RGB_result


def rmse(apath,bpath):
    """
    This is the help function to get RMSE score.
    apath: path to your result
    bpath: path to our reference image
    when saving your result to disk, please clip it to 0,255:
    .clip(0.0, 255.0).astype(np.uint8))
    """
    a = cv2.imread(apath).astype(np.float32)
    b = cv2.imread(bpath).astype(np.float32)
    print(np.sqrt(np.mean((a-b)**2)))


if __name__ == "__main__":
    print('==================================================')
    
    path_file_image_source = sys.argv[1]
    path_file_image_target = sys.argv[2]
    path_file_image_result_in_Lab = sys.argv[3]

    # todo: read input images
    # img_RGB_source: is the image you want to change the its color
    # img_RGB_target: is the image containing the color distribution that you want to change the img_RGB_source to (transfer color of the img_RGB_target to the img_RGB_source)
    # OpenCv read an image in BGR format. So we need to implement convert_color_space_BGR_to_RGB() to get the RGB format

    img_BGR_source = cv2.imread(path_file_image_source)
    img_BGR_target = cv2.imread(path_file_image_target)
    
    img_RGB_source = convert_color_space_BGR_to_RGB(img_BGR_source)
    img_RGB_target = convert_color_space_BGR_to_RGB(img_BGR_target)
    
    img_RGB_result = color_transfer(img_RGB_source, img_RGB_target)
    img_BGR_result = convert_color_space_RGB_to_BGR(img_RGB_result).clip(0.0, 255.0).astype(np.uint8)
    cv2.imwrite(path_file_image_result_in_Lab, img_BGR_result)

    # todo: save image to path_file_image_result_in_Lab