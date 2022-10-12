from keras import backend as K
from keras.utils import np_utils
import tensorflow as tf
import cv2
import os
import numpy as np
from itertools import product
import itertools


############ Data generation ##########
def standarization(x_train):
    mean, std = x_train.mean(), x_train.std()
    # global standardization of pixels
    pixels = (x_train - mean) / std
    # clip pixel values to [-1,1]
    pixels = np.clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0
    return pixels


def remove_files(path):
    for root, directory, files in os.walk(path):
        for fname in files:
            os.remove(os.path.join(root, fname))
    return 0

def array2img(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return x

def adjust_gamma(image, gamma=0.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

    
def square_from_center(image, center_y, center_x, window_size):
    origin_y = int(center_y - (window_size - 1) / 2)
    origin_x = int(center_x - (window_size - 1) / 2)
    return np.array(image[origin_y:origin_y + window_size, origin_x:origin_x + window_size])

def square_from_center_label(image, center_y, center_x, window_size):
    origin_y = int(center_y - (window_size - 1) / 2)
    origin_x = int(center_x - (window_size - 1) / 2)
    return np.array(image[origin_y:origin_y + window_size, origin_x:origin_x + window_size]).astype(np.float64)

def generate_data_from_center_coords(image, coordinates, window_size, gamma=1.0):
    data = []
    idx = []
    for y_coord, x_coord in coordinates:
        cropped_image = square_from_center(image, y_coord, x_coord, window_size)
        if cropped_image.shape != (window_size, window_size, 3):
            continue
        else:
            if gamma == 1.0:
                data.append(cropped_image) 
            else:
                data.append(adjust_gamma(cropped_image, gamma))
            idx.append([y_coord, x_coord])
    return np.array(data), np.array(idx)

def generate_data_from_center_coords_label(image, coordinates, window_size):
    data = []

    for y_coord, x_coord in coordinates:
        cropped_image = square_from_center_label(image, y_coord, x_coord, window_size)
#         print('single label size: ', cropped_image.shape)
        if cropped_image.shape != (window_size, window_size, 3):
            continue
        else:
            data.append(cropped_image)
    return np.array(data)

def label_generation(image, window_size, NB_CLASSES):
    tmpt = np.array(image[:,:,0]).astype(np.float32)
    label = np.zeros((window_size, window_size, 2))
    for row in range(tmpt.shape[0]):
        label[row, :] = np_utils.to_categorical(np.array(tmpt[row, :]), num_classes=NB_CLASSES)
        #label[row, :] = np_utils.to_categorical(np.zeros((128,1)), num_classes=NB_CLASSES)
    return label


def points_generator(DATA_PATH, OBJECT_LIST, OBJECT_NUMS, MAP_PATH, random=True):
    obj_list = []
    obj_length = len(OBJECT_LIST)
        
    for obj_index in range(obj_length):
        obj_name = OBJECT_LIST[obj_index]
        print('read points from file: ', os.path.join(DATA_PATH, obj_name+'.txt'))
        obj_points = np.loadtxt(os.path.join(DATA_PATH, obj_name+'.txt'), dtype=np.int32, delimiter=",")
        if obj_points.size == 2: # if txt only has one point, extend the dim
            obj_points = np.expand_dims(obj_points, 0)
        np.random.shuffle(obj_points)
        obj_list.append(obj_points)
    ##### load the positive coord in the first txt 
    x_train_coor_pos = obj_list[0][:OBJECT_NUMS[0]]

    x_train_coor_neg = []
    if obj_length > 2: # if length=2, 1st num is #positive samples, 2nd num is #negative samples
        for i in range(1, obj_length):
            if i == 1:
                x_train_coor_neg = obj_list[i][:OBJECT_NUMS[i]]
            else:
                x_train_coor_neg = np.concatenate((x_train_coor_neg, obj_list[i][:OBJECT_NUMS[i]]), axis=0)
        print('negative points from other categories: ', x_train_coor_neg.shape)
    
    if random and OBJECT_NUMS[-1]>0:
        dis_thres = 200 ### the distance btw rand coord and positive coord
        img = cv2.imread(MAP_PATH)
        p_rand = []
        height, width = img.shape[0], img.shape[1]
        x_rand = np.random.randint(height, size=OBJECT_NUMS[-1])
        y_rand = np.random.randint(width, size=OBJECT_NUMS[-1])
        for i in range(OBJECT_NUMS[-1]):
            for j in range(obj_list[0].shape[0]):
                if np.linalg.norm([x_rand[i],y_rand[i]]-obj_list[0][j]) < dis_thres:
                    continue
            p_rand.append([x_rand[i],y_rand[i]])
        p_rand = np.array(p_rand)
        print('random points: ', p_rand.shape)
        if x_train_coor_neg != []:
            x_train_coor_neg = np.concatenate((x_train_coor_neg, p_rand), axis=0)
        else:
            x_train_coor_neg = p_rand
    return x_train_coor_pos, x_train_coor_neg


def data_generator(DATA_PATH, MAP_PATH, LABEL_PATH, OBJECT_LIST, OBJECT_NUMS, WIN_SIZE, NB_CLASSES,\
                   shift_augment=False, shift_range=(0, 0), num_shift=2, times4multi=1,\
                   check_data=False, gamma=1.0, random=True):
    img = cv2.imread(MAP_PATH)
    label = cv2.imread(LABEL_PATH)
    pos_coor, neg_coor =  points_generator(DATA_PATH,OBJECT_LIST,OBJECT_NUMS,MAP_PATH,random=random)
    print('Before augmentation, total positive, negative coor: ', pos_coor.shape, neg_coor.shape)
    if shift_augment == False:
        coor = np.vstack((pos_coor, neg_coor))
        np.random.shuffle(coor)
    else:
        aug_pos_coor = gen_shifted_coor(pos_coor, shift_range, num_shift=num_shift,\
                                        times4multi=times4multi, label_img=label) # num_shift=2 means doubling the data
        print('after shifting positive coor shape: ', aug_pos_coor.shape)
        coor = np.vstack((aug_pos_coor, neg_coor))
        np.random.shuffle(coor)
        print('after shiting all coor shape: ', coor.shape)
    x_train, x_indices = generate_data_from_center_coords(img, coor, WIN_SIZE, gamma)
    print('x_train shape: ', x_train.shape)

    y_train = generate_data_from_center_coords_label(label, coor, WIN_SIZE)
    print('y_train shape: ', y_train.shape)

    y_train = np.array(y_train)
    x_train = x_train.astype(np.float64)
        #x_train = standarization(x_train)
#         x_train = preprocess_input(x_train)
#     else:
#         x_train, y_train = data_augmentation_shuffle(img, label, pos_coor, neg_coor, WIN_SIZE, NB_CLASSES,gamma)
    
    remove_files('../data/train')
    remove_files('../data/train_labels')

    return x_train, y_train, x_indices

def rotation(img, angle):
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (h, w))
    return rotated

def gen_shifted_coor(coor_arr, shift_range, num_shift=2, times4multi=1, label_img=None):
    ##### coor_arr: (#coords, 2), a list of coordinates
    ##### shift range: (dx, dy), the shift is (-dx, dx), (-dy, dy)
    ##### num_shift: randomly select how many (dx, dy) pairs
    ##### return augmented coor_arr (#aug_coords, 2)
    aug_coor_arr = []
    for x, y in coor_arr:
        dx = np.random.randint(-shift_range[0], shift_range[0], (num_shift))
        dy = np.random.randint(-shift_range[1], shift_range[1], (num_shift))
        aug_coor_arr.append([x, y])
        for i in range(num_shift):
            new_x, new_y = x+dx[i], y+dy[i]
            aug_coor_arr.append([new_x, new_y])
        if times4multi != 1:
            label_list = generate_data_from_center_coords_label(label_img, [[new_x,new_y]], 256)
            if len(label_list) == 0:
                continue
            label = label_list[0]
            if np.sum(label[:,:,1]/255) > 270:
                more_dx = np.random.randint(-shift_range[0], shift_range[0], (num_shift*times4multi))
                more_dy = np.random.randint(-shift_range[1], shift_range[1], (num_shift*times4multi))
                for j in range(num_shift*times4multi):
                    more_new_x, more_new_y = x+more_dx[j], y+more_dy[j]
                    aug_coor_arr.append([more_new_x, more_new_y])
    aug_coor_arr = np.array(aug_coor_arr)
    return aug_coor_arr

def data_augmentation(img, label, pos_coor, neg_coor, WIN_SIZE, NB_CLASSES, gamma=1.0):
    x_train_pos = generate_data_from_center_coords(img, pos_coor, WIN_SIZE, gamma)
    y_train_pos = generate_data_from_center_coords_label(label, pos_coor, WIN_SIZE)

    x_train_neg = generate_data_from_center_coords(img, neg_coor, WIN_SIZE, gamma)
    y_train_neg = generate_data_from_center_coords_label(label,neg_coor, WIN_SIZE)
    x_train_expanded, y_train_expanded = [], []
    
    x_pos_rotate, y_pos_rotate = [], []
    for i in range(x_train_pos.shape[0]):
        for angle in range(0, 360, 90):
            x_temp, y_temp = rotation(x_train_pos[i], angle), rotation(y_train_pos[i], angle)
            x_pos_rotate.append(x_temp)
            y_pos_rotate.append(y_temp)
            x_train_expanded.append(x_temp)
            y_train_expanded.append(y_temp)
                                    
    bright_seeds = np.random.uniform(0.5,1.5,len(x_pos_rotate))
    x_pos_bright, y_pos_bright = [], []
    for i in range(bright_seeds.shape[0]):
        x_temp = adjust_gamma(x_pos_rotate[i], bright_seeds[i])
        x_pos_bright.append(x_temp)
        y_pos_bright.append(y_pos_rotate[i])
        x_train_expanded.append(x_temp)
        y_train_expanded.append(y_pos_rotate[i])
    
    x_neg_rotate, y_neg_rotate = [], []
    for i in range(x_train_neg.shape[0]):
        for angle in range(0, 360, 90):
            x_temp, y_temp = rotation(x_train_neg[i], angle), rotation(y_train_neg[i], angle)
            x_neg_rotate.append(x_temp)
            y_neg_rotate.append(y_temp)
            x_train_expanded.append(x_temp)
            y_train_expanded.append(y_temp)
    
    bright_seeds = np.random.uniform(0.5,1.5,len(x_neg_rotate))
    x_neg_bright, y_neg_bright = [], []
    for i in range(bright_seeds.shape[0]):
        x_temp = adjust_gamma(x_neg_rotate[i], bright_seeds[i])
        x_neg_bright.append(x_temp)
        y_neg_bright.append(y_neg_rotate[i])
        x_train_expanded.append(x_temp)
        y_train_expanded.append(y_neg_rotate[i])

    x_train_expanded, y_train_expanded = np.array(x_train_expanded), np.array(y_train_expanded)  
#     x_train_expanded = x_train_expanded.astype(np.float64)
    print('x_expand, y_expand: ', x_train_expanded.shape, y_train_expanded.shape)
#     x_train_expanded = preprocess_input(x_train_expanded)
    #x_train_expanded = standarization(x_train_expanded)
    annotation_train = []
    for i in range(y_train_expanded.shape[0]):
        annotation_train.append(label_generation(y_train_expanded[i,:,:,:], WIN_SIZE, NB_CLASSES))
    annotation_train = np.array(annotation_train)
    return x_train_expanded, annotation_train

def data_augmentation_shuffle(img, label, pos_coor, neg_coor, WIN_SIZE, NB_CLASSES,gamma=1.0):
    coor = np.concatenate((pos_coor, neg_coor), axis=0)
    np.random.shuffle(coor)
    x_train = generate_data_from_center_coords(img, coor, WIN_SIZE, gamma)
    y_train = generate_data_from_center_coords_label(label, coor, WIN_SIZE)

    x_train_expanded, y_train_expanded = [], []
    
    x_rotate, y_rotate = [], []
    for i in range(x_train.shape[0]):
        for angle in range(0, 360, 90):
            x_temp, y_temp = rotation(x_train[i], angle), rotation(y_train[i], angle)
            x_rotate.append(x_temp)
            y_rotate.append(y_temp)
            x_train_expanded.append(x_temp)
            y_train_expanded.append(y_temp)
                                    
    bright_seeds = np.random.uniform(0.5,1.5,len(x_rotate))
    x_bright, y_bright = [], []
    for i in range(bright_seeds.shape[0]):
        x_temp = adjust_gamma(x_rotate[i], bright_seeds[i])
        x_bright.append(x_temp)
        y_bright.append(y_rotate[i])
        x_train_expanded.append(x_temp)
        y_train_expanded.append(y_rotate[i])

    x_train_expanded, y_train_expanded = np.array(x_train_expanded), np.array(y_train_expanded)  
#     x_train_expanded = x_train_expanded.astype(np.float64)
    print('x_expand, y_expand: ', x_train_expanded.shape, y_train_expanded.shape)
#     x_train_expanded = preprocess_input(x_train_expanded)
    #x_train_expanded = standarization(x_train_expanded)
#     annotation_train = []
#     for i in range(y_train_expanded.shape[0]):
#         annotation_train.append(label_generation(y_train_expanded[i,:,:,:], WIN_SIZE, NB_CLASSES))
#     annotation_train = np.array(annotation_train)
    return x_train_expanded, y_train_expanded