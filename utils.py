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
    image = np.transpose(image, (0, 2, 3, 1))[0]
    # img = Image.fromarray(image[0], mode='RGB')
    # gray = img.convert('LA')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rectangles = detector(gray.astype(np.uint8), 2)
    if len(face_rectangles) == 0:
        return None
    largest_face_rectangle = find_largest_face(face_rectangles)

    return [largest_face_rectangle.left(), largest_face_rectangle.top(),
            largest_face_rectangle.right(), largest_face_rectangle.bottom()]


def grab_face(frame):
    w = 480
    h = 640
    good_shape = (256, 256, 3)

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

        image = np.transpose(frame[0], (1, 2, 0)).astype(np.uint8)
        img = Image.fromarray(image, mode='RGB')
        img = img.crop(optface)  # left, upper, right, and lower
        # img.save('eee.jpg')
        # save image to see if good
        # img.save('/home/gabras/deployed/deepimpression2/chalearn30/crops/crop_bg.jpg')
        img = np.array(img)

        # if image is not square, fill bottom with mean of face
        if img.shape != good_shape:
            px_mean = np.sum(img, 0)
            px_mean = np.sum(px_mean, 0)
            px_mean /= (img.shape[0] * img.shape[1])

            # px_mean = np.mean(img, 2)
            # px_mean = np.mean(px_mean, 2)

            canvas = np.ones(good_shape, dtype=img.dtype) * px_mean.astype(np.uint8)

            img2 = Image.fromarray(img, mode='RGB')

            if img.shape[0] > img.shape[1]:
                resize_factor = 256. / img.shape[0]
                n_h = int(img.shape[1] * resize_factor) - 1
                img = img2.resize((256, n_h))
                img = np.array(img)
            else:
                resize_factor = 256. / img.shape[1]
                n_w = int(img.shape[0] * resize_factor) - 1
                img = img2.resize((n_w, 256))
                img = np.array(img)

            try:
                canvas[0:img.shape[0], 0:img.shape[1]] = img
            except ValueError:
                print('hmm')

            img = canvas.astype(np.uint8)

        image = np.transpose(img, (2, 0, 1))
        image = np.expand_dims(image, 0)
    else:
        # just black image
        image = np.zeros(good_shape, dtype=frame.dtype)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        print('no face')

    return image, optface


def predict_frame(data, model):
    data, bb = grab_face(data)
    data = data.astype(np.float32)
    with chainer.using_config('train', False):
        p = model(data)
        print(p)
    return p, bb


def draw_bb(frame, optface):
    # left, up, right, down = optface
    up, left, down, right = optface
    up -= 1
    left -= 1
    down -= 1
    right -= 1
    print(optface)
    print(frame.shape)

    try:
        # frame[up, left:right] = [0, 255, 0]
        # frame[down, left:right] = [0, 255, 0]
        # frame[up:down, left] = [0, 255, 0]
        # frame[up:down, right] = [0, 255, 0]
        frame[left:right, up] = [0, 255, 0]
        frame[left:right, down] = [0, 255, 0]
        frame[left, up:down] = [0, 255, 0]
        frame[right, up:down] = [0, 255, 0]
    except IndexError:
        print('hm')

    return frame