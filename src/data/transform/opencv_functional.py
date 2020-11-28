'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-28 07:36:15
'''

import numpy as np
import cv2 as cv
import torch
import warnings
import numbers
from PIL import Image
from PIL import ImageFilter,ImageEnhance

def _is_tensor_image(image):
    '''
    Description:  Return whether image is torch.tensor and the number of dimensions of image.
    Reutrn : True or False.
    '''
    return torch.is_tensor(image) and image.ndimension()==3

def _is_numpy_image(image):
    '''
    Description: Return whether image is np.ndarray and the number of dimensions of image
    Return: True or False.
    '''
    return isinstance(image,np.ndarray) and (image.ndim in {2,3} )

def _is_numpy(landmarks):
    '''
    Description: Return whether landmarks is np.ndarray.
    Return: True or False
    '''
    return isinstance(landmarks,np.ndarray)

def to_tensor(sample):
    '''
    Description: Convert sample.values() to Tensor.
    Args (type): sample : {image:np.ndarray,mask:np.ndarray}
    Return: Converted sample
    '''
    image = sample['image']
    # _check
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be numpy.ndarray. Got {}".format(type(image)))
        
    # handle numpy.array
    if image.ndim == 2:
        image = image[:,:,None]

    # Swap color axis because 
    # numpy image: H x W x C
    # torch image: C x H x W 
    image = torch.from_numpy(image.transpose((2,0,1)))
    if isinstance(image,torch.ByteTensor) or image.dtype == torch.uint8:
        image = image.float().div(255)
    
    sample['image'] = image
    
    if "mask" in sample:
        mask = sample["mask"]
        if not(_is_numpy_image(mask)):
            raise TypeError("sample['mask'] should be numpy.ndarray. Got {}".format(type(mask)))
        if mask.ndim == 2:
            mask = mask[:,:,None]
        mask  = torch.from_numpy(mask.transpose((2,0,1)))
        mask = mask.float()
        sample['mask'] = mask
    return sample


def normalize(sample,mean,std,inplace=False):
    '''
    Description: Normalize a tensor image with mean and standard deviation.
    Args (type): 
        sample (dict): 
            sample({"image":image,"mask":mask})
        mean (sequnence): Sequence of means for each channel.
        std (sequence): Sequence of standard devication for each channel.
    Return: 
        Converted sample
    '''
    image = sample['image']
    if not _is_tensor_image(image):
        raise TypeError("image should be a torch image. Got {}".format(type(image)))

    if not inplace:
        image = image.clone()

    # check dtype and device 
    dtype = image.dtype
    device = image.device
    mean = torch.as_tensor(mean,dtype=dtype,device=device)
    std = torch.as_tensor(std,dtype=dtype,device=device)
    image.sub_(mean[:,None,None]).div_(std[:,None,None])
    sample['image'] = image
    return sample


def hflip(sample):
    '''
    Description: Horizontally flip the given sample
    Args (type): 
        sample (dict):
            sample ({"image":image,"mask":mask})
    Return: 
        Converted sample
    '''
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

    if image.shape[2] == 1:
        image = cv.flip(image,1)[:,:,np.newaxis] #keep image.shape = H x W x 1
    else:
        image = cv.flip(image,1)
    sample['image'] = image

    if 'mask' in sample:
        mask = sample['mask']
        if not _is_numpy_image(mask):
            raise TypeError("sample['mask'] should be np.ndarray image. Got {}".format(type(mask)))
        mask = cv.flip(mask,1)
        sample['mask'] = mask
    return sample
    
def vflip(sample):
    '''
    Description: Vertically flip the given sample
    Args (type): 
        sample (dict):
            sample ({"image":image,"mask":mask})
    Return: 
        Converted sample
    '''
    image = sample['image']
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))

    if image.shape[2] == 1:
        image = cv.flip(image,0)[:,:,np.newaxis] #keep image.shape = H x W x 1
    else:
        image = cv.flip(image,0)
    sample['image'] = image

    if 'mask' in sample:
        mask = sample['mask']
        if not _is_numpy_image(mask):
            raise TypeError("sample['mask'] should be np.ndarray image. Got {}".format(type(mask)))
        mask = cv.flip(mask,0)
        sample['mask'] = mask
    return sample

def randomcrop(sample,output_size):
    image = sample['image']    
    if not _is_numpy_image(image):
        raise TypeError("sample['image'] should be np.ndarray image. Got {}".format(type(image)))
    image_h,image_w = image.shape[0],image.shape[1]
    crop_w,crop_h = output_size
    assert (image_w>crop_w) and (image_h>crop_h)
    
    topleft_h = np.random.randint(low=0,high=image_h-crop_h)
    topleft_w = np.random.randint(low=0,high=image_w-crop_w)
    crop_image = image[topleft_h:topleft_h+crop_h,topleft_w:topleft_w+crop_w,:]
    sample['image'] = crop_image

    if 'mask' in sample:
        mask = sample['mask']
        if not _is_numpy_image(mask):
            raise TypeError("sample['mask'] should be np.ndarray image. Got {}".format(type(mask)))
        crop_mask = mask[topleft_h:topleft_h+crop_h,topleft_w:topleft_w+crop_w]
        sample['mask'] = crop_mask
    return sample

def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        numpy ndarray: Brightness adjusted image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ i*brightness_factor for i in range (0,256)]).clip(0,255).astype('uint8')
    # same thing but a bit slower
    # cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
    if img.shape[2]==1:
        return cv.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv.LUT(img, table)

def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an mage.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        numpy ndarray: Contrast adjusted image.
    """
    # much faster to use the LUT construction than anything else I've tried
    # it's because you have to change dtypes multiple times
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    table = np.array([ (i-74)*contrast_factor+74 for i in range (0,256)]).clip(0,255).astype('uint8')
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(contrast_factor)
    if img.shape[2]==1:
        return cv.LUT(img, table)[:,:,np.newaxis]
    else:
        return cv.LUT(img,table)

def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        numpy ndarray: Saturation adjusted image.
    """
    # ~10ms slower than PIL!
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return np.array(img)

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (numpy ndarray): numpy ndarray to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        numpy ndarray: Hue adjusted image.
    """
    # After testing, found that OpenCV calculates the Hue in a call to 
    # cv2.cvtColor(..., cv2.COLOR_BGR2HSV) differently from PIL

    # This function takes 160ms! should be avoided
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))
    if not _is_numpy_image(img):
        raise TypeError('img should be numpy Image. Got {}'.format(type(img)))
    img = Image.fromarray(img)
    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return np.array(img)

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return np.array(img)



        
