from model import Deepimpression
import chainer
import numpy as np
import dlib
import cv2
from PIL import Image


def load_model():
    model = Deepimpression()
    # load weights from epoch_29_34
    p = '/home/gabi/PycharmProjects/visualizing-traits/src/standalone_demo/media/epoch_29_34'
    chainer.serializers.load_npz(p, model)
    return model


def find_largest_face(face_rectangles):
    number_rectangles = len(face_rectangles)

    if number_rectangles == 0:
        return None
    elif number_rectangles == 1:
        return face_rectangles[0]
    else:
        largest = 0
        which_rectangle = None
        for i in range(number_rectangles):
            r = face_rectangles[i]
            # it's a square so only one side needs to be checked
            width = r.right() - r.left()
            if width > largest:
                largest = width
                which_rectangle = i
        return face_rectangles[which_rectangle]


def find_face_simple(image):
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rectangles = detector(gray, 2)
    if len(face_rectangles) == 0:
        return None
    largest_face_rectangle = find_largest_face(face_rectangles)

    return [largest_face_rectangle.left(), largest_face_rectangle.top(),
            largest_face_rectangle.right(), largest_face_rectangle.bottom()]


def grab_face(frame):
    w = 456
    h = 256
    good_shape = (w, h, 3)

    optface = find_face_simple(frame)
    if optface is not None:
        if optface[3] > h:
            optface[3] = h
        if optface[2] > w:
            optface[2] = w
        if optface[1] < 0:
            optface[1] = 0
        if optface[0] < 0:
            optface[0] = 0

    image = np.transpose(frame[0], (1, 2, 0))
    img = Image.fromarray(image, mode='RGB')
    img = img.crop(optface)  # left, upper, right, and lower
    # save image to see if good
    # img.save('/home/gabras/deployed/deepimpression2/chalearn30/crops/crop_bg.jpg')
    img = np.array(img)
    
    # if image is not square, fill bottom with mean of face
    if img.shape != good_shape:
        px_mean = np.mean(img, 2)
        px_mean = np.mean(px_mean, 2)

        canvas = np.ones(good_shape, dtype=img.dtype) * px_mean
        canvas[0:img.shape[0], 0:img.shape[1]] = img

        img = canvas

    image = np.transpose(img, (2, 0, 1))
    image = np.expand_dims(image, 0)

    return image


def predict_frame(data, model):
    data = grab_face(data)
    with chainer.using_config('train', False):
        p = model(data)
    return p