import cv2
import numpy as np
import random


def invert(image):
    return 255 - image


def blur3(image):
    return cv2.blur(image, (3, 3))


def blur5(image):
    return cv2.blur(image, (5, 5))

def sq_blur(image):
    r = 0.8
    h = image.shape[0]
    w = image.shape[1]
    h_r = int(h*r)
    w_r = int(w*r)
    image = cv2.resize(image, (h_r, w_r), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (h, w), interpolation=cv2.INTER_AREA)
    return image


def random_brightness(image):
    c = random.uniform(0.2, 1.8)
    blank = np.zeros(image.shape, image.dtype)
    dst = cv2.addWeighted(image, c, blank, 1 - c, 0)
    return dst


def rotate(image, scale=1.0):
    angle = random.uniform(-5, 5)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



def gray_scale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.merge((gray, gray, gray))
    return dst



cv2aug_tuple = (invert,  sq_blur, random_brightness,
                    rotate,  gray_scale, blur3, blur5)

cv2aug_dict = {corr_func.__name__: corr_func for corr_func in
                   cv2aug_tuple}


def cv2aug(image, aug_number=-1,  aug_name=None):
    """This function returns a corrupted version of the given image.

    Args:
        image (numpy.ndarray):      image to corrupt; a numpy array in [0, 255], expected datatype is np.uint8
                                    expected shape is either (height x width x channels) or (height x width);
                                    width and height must be at least 32 pixels;
                                    channels must be 1 or 3;
        severity (int):             strength with which to corrupt the image; an integer in [1, 5]
        corruption_name (str):      specifies which corruption function to call, must be one of
                                        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                                        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                                        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                                    the last four are validation corruptions
        corruption_number (int):    the position of the corruption_name in the above list; an integer in [0, 18];
                                        useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    Returns:
        numpy.ndarray:              the image corrupted by a corruption function at the given severity; same shape as input
    """

    if not isinstance(image, np.ndarray):
        raise AttributeError('Expecting type(image) to be numpy.ndarray')
    if not (image.dtype.type is np.uint8):
        raise AttributeError('Expecting image.dtype.type to be numpy.uint8')

    if not (image.ndim in [2, 3]):
        raise AttributeError('Expecting image.shape to be either (height x width) or (height x width x channels)')
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)

    height, width, channels = image.shape

    if (height < 32 or width < 32):
        raise AttributeError('Image width and height must be at least 32 pixels')

    if not (channels in [1, 3]):
        raise AttributeError('Expecting image to have either 1 or 3 channels (last dimension)')

    if channels == 1:
        image = np.stack((np.squeeze(image),) * 3, axis=-1)


    elif aug_number != -1:
        image_corrupted = cv2aug_tuple[aug_number](image)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(image_corrupted)


class Augment():
    def __init__(self):
        self.inv_prob = 0.5
        self.blur_prob = 0.3
        self.sq_blur_prob = 0.3
        self.bright_prob = 0.5
        self.rotate_prob = 1.
        self.zoom_prob = 1.
        self.gray_prob = 0.0

    def invert(self, image):
        return 255 - image

    def blur(self, image):
        return cv2.blur(image, (3, 3))

    def sq_blur(self, image):
        image = cv2.resize(image, (400, 128), interpolation=cv2.INTER_AREA)
        return image

    def random_brightness(self, image):
        c = random.uniform(0.2, 1.8)
        blank = np.zeros(image.shape, image.dtype)
        dst = cv2.addWeighted(image, c, blank, 1 - c, 0)
        return dst

    def rotate(self, image, scale=1.0):
        angle = random.uniform(-5, 5)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def zoom(self, image, scale=0.3):
        h, w = image.shape[:2]
        w_ = int(w * 128 / h)
        if w_ > 400:
            return image
        else:
            w_ = random.randint(max(1, int(w_ * (1 - scale))), w_)
            image = cv2.resize(image, (w_, 128))
            return image

    def gray_scale(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.merge((gray, gray, gray))
        return dst

    def apply(self, image):
        inv_prob = random.random()
        blur_prob = random.random()
        sq_blur_prob = random.random()
        bright_prob = random.random()
        rotate_prob = random.random()
        zoom_prob = random.random()

        if inv_prob < self.inv_prob:
            image = self.invert(image)

        if bright_prob < self.bright_prob:
            image = self.random_brightness(image)

        if rotate_prob < self.rotate_prob:
            image = self.rotate(image)

        if zoom_prob < self.zoom_prob:
            image = self.zoom(image)

        if blur_prob < self.blur_prob:
            image = self.blur(image)

        if sq_blur_prob < self.sq_blur_prob:
            image = self.sq_blur(image)

        return image

