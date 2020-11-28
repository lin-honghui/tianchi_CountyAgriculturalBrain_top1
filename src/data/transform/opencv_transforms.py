'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-28 06:13:56
'''
import math
import torch
import types
import numbers
import warnings
import cv2 as cv
import numpy as np
from . import opencv_functional as F
from PIL import ImageFilter,ImageEnhance







class RandomChoice(object):
    """
    Apply transformations randomly picked from a list with a given probability
    Args:
        transforms: a list of transformations
        p: probability
    """
    def __init__(self,p,transforms):
        self.p = p
        self.transforms = transforms
    def __call__(self,sample):
        if len(self.transforms) < 1:
            raise TypeError("transforms(list) should at least have one transformation")
        for t in self.transforms:
            if np.random.uniform(0,1) < self.p:
                sample = t(sample)
        return sample

    def __repr__(self):
        return self.__class__.__name__+"(p={})".format(self.p)

class Compose(object):
    '''
    Description: Compose several transforms together
    Args (type): 
        transforms (list): list of transforms
        sample (ndarray or dict):
    return: 
        sample (ndarray or dict)
    '''
    def __init__(self,transforms):
        self.transforms = transforms
    def __call__(self,sample):
        for t in self.transforms:
            sample = t(sample)
        return sample
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    '''
    Description: Convert ndarray in sample to Tensors.
    Args (type): 
        sample (ndarray or dict)
    return: 
        Converted sample.
    '''
    def __call__(self,sample):
        return F.to_tensor(sample)
    def __repr__(self):
        return self.__class__.__name__ + "()"

class Normalize(object):
    '''
    Description: Normalize a tensor with mean and standard deviation.
    Args (type): 
        mean (tuple): Sequence of means for each channel.
        std (tuple): Sequence of std for each channel.
    Return: 
        Converted sample
    '''
    def __init__(self,mean,std,inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self,sample):
        #Convert to tensor
        mean = torch.tensor(self.mean,dtype=torch.float32)
        std = torch.tensor(self.std,dtype=torch.float32)
        return F.normalize(sample,mean,std,inplace=self.inplace)
    def __repr__(self):
        format_string = self.__class__.__name__ + "(mean={0},std={1})".format(self.mean,self.std)
        return format_string

class RandomHorizontalFlip(object):
    '''
    Description: Horizontally flip the given sample with a given probability.
    Args (type): 
        p (float): probability of the image being flipped. Default value is 0.5.
    Return: Converted sample
    '''
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.hflip(sample)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip(object):
    '''
    Description: Vertically flip the given sample with a given probability.
    Args (type): 
        p (float): probability of the image being flipped. Default value is 0.5.
    Return: 
        Converted sample
    '''
    def __init__(self,p=0.5):
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.vflip(sample)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)

        
class Lambda(object):
    '''
    Description: Apply a user-defined lambda as a transform.
    Args (type): lambd (function): Lambda/function to be used for transform.
    Return: 
    '''
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ColorJitter(object):
    '''
    Description: Randomly change the brightness, contrast and saturation of an image.
    Args (type): 
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    Return: 
        Converted sample
    '''
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        if self.saturation is not None:
            warnings.warn('Saturation jitter enabled. Will slow down loading immensely.')
        if self.hue is not None:
            warnings.warn('Hue jitter enabled. Will slow down loading immensely.')
    
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = np.random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = np.random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = np.random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = np.random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        """
        Args:
            img (numpy ndarray): Input image.
        Returns:
            numpy ndarray: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        image = sample['image']
        if isinstance(image,np.ndarray) and image.ndim in {2,3}:
            image = transform(image)
            sample['image'] = image
            return sample
        else:
            raise TypeError("Image should be a numpu.ndarray image. Got {}".format(type(image)))

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RandomCrop(object):
    '''
    Description:  Crop randomly the image in a sample
    Args (type): 
        output_size(tuple or int):Desized output size.
        If int,square crop is made
    Return: 
    '''
    def __init__(self,p,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        self.p = p
    def __call__(self,sample):
        if np.random.random() < self.p:
            return F.randomcrop(sample,self.output_size)
        else:
            return sample
    def __repr__(self):
        return self.__class__.__name__ + '(output_size={})'.format(self.output_size)
    


