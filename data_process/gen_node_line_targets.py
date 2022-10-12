###### only process the single lines
from util.process_shp import read_shp, interpolation
from data_loader import data_generator, remove_files
import os, cv2, math
import numpy as np
from data_process.gen_line_list import gen_line_list
from data_process.gen_single_line_target import gen_single_line_target

def rotate(image, angle=90):
    row,col = image.shape[:2]
    center = tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def rotate_pt(point, cent, angle_degree=90):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle in [0, 360].
    """
    angle = math.radians(angle_degree)
    s, c = math.sin(angle), math.cos(angle)
#      translate point back to origin:
    px, py = point
    cx, cy = cent
    px -= cx
    py -= cy
#    rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c
#    translate point back:
    px = xnew + cx
    py = ynew + cy
    return int(abs(px)), int(abs(py))

def rotate_line_list(line_list, angle_degree=0):
    rot_line_list = []
    for ln in line_list:
        one_rot_line = []
        for pt in ln:
            rot_pt = rotate_pt(pt, (128,128), angle_degree)
            one_rot_line.append([rot_pt[0], rot_pt[1]])
        rot_line_list.append(one_rot_line)
    return rot_line_list

def gen_target_nodes_edges(data_dir, png_map_name, png_label_name, tif_map_name, shp_label_name, OBJECT_LIST, OBJECT_NUMS,\
                           grid=32, img_size=256, num_dec_nodes=32, shift_augment=True, shift_range=(20, 20), num_shift=3):

    map_path = os.path.join(data_dir, png_map_name)
    label_path = os.path.join(data_dir, png_label_name) 

    WIN_SIZE = img_size
    NB_CLASSES = 2
    gamma = 1.0

    shp_path = os.path.join(data_dir, shp_label_name)
    tif_path = os.path.join(data_dir, tif_map_name)

    polylines = read_shp(shp_path, tif_path)
    
    polylines_interp = []
    inter_dis = 10

    for i, line in enumerate(polylines):
        for p in range(len(line)-1):
            x_s, y_s = line[p]
            x_e, y_e = line[p+1]
            vec_interp = interpolation([x_s, y_s], [x_e, y_e], inter_dis)
            if vec_interp == None:
                continue
            polylines_interp.append(vec_interp)

    all_x_train, all_y_train, all_img_indices = \
        data_generator(data_dir, map_path, label_path, OBJECT_LIST, OBJECT_NUMS, WIN_SIZE, NB_CLASSES, \
                       shift_augment=shift_augment, shift_range=shift_range, num_shift=num_shift, times4multi=1,\
                       gamma=gamma, random=True)
    
    x_train, y_train, img_indices = [], [], []
    for i, x_img in enumerate(all_x_train):
        if (all_y_train[i,:,:,0]/255).sum() < 310 or (all_y_train[i,:,:,0]/255).sum()==0:
            x_train.append(all_x_train[i])
            y_train.append(all_y_train[i])
            img_indices.append(all_img_indices[i])
    print('After removing the multilines images, num of x_train shape: ', len(x_train))        
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    img_indices = np.array(img_indices)
    
    all_enc_cat_targets, all_enc_reg_targets, all_enc_mask = [], [], []
    all_dec_inputs, all_dec_targets, all_dec_mask = [], [], []
    
    bd_coord_list, rot_x_train = [], []
    for i, x_img in enumerate(x_train):
        xmin, ymin, xmax, ymax = img_indices[i][0]-WIN_SIZE//2, img_indices[i][1]-WIN_SIZE//2, \
                                img_indices[i][0]+WIN_SIZE//2, img_indices[i][1]+WIN_SIZE//2

        bd_coord_list.append([xmin, ymin, xmax, ymax])
        
    for i, bd_coord in enumerate(bd_coord_list):
        orig_line_list = gen_line_list(polylines_interp, bd_coord[0], bd_coord[1], bd_coord[2], bd_coord[3])

        ngrids = int((img_size//grid)**2)
        
        for r_angle in [90, 180, 270, 360]:
            img_rotate = rotate(x_train[i], angle=r_angle)
            rot_x_train.append(img_rotate)
            line_list = rotate_line_list(orig_line_list, angle_degree=r_angle)
            enc_cat_targets = np.zeros((ngrids))
            enc_reg_targets = np.ones((ngrids, 2))
            enc_mask = np.ones((ngrids))
            dec_inputs = np.ones((num_dec_nodes, 2))
            dec_targets = np.ones((num_dec_nodes))*(num_dec_nodes-1)
            dec_mask = np.ones((num_dec_nodes))

            if len(line_list) > 0: ##### only process signle line
                lens = np.array([len(l) for l in line_list])
                max_id = np.argmax(lens)
                enc_cat_targets, enc_reg_targets, enc_mask, dec_inputs, dec_targets, dec_mask = \
                    gen_single_line_target(line_list, enc_cat_targets, enc_reg_targets, enc_mask, dec_inputs,\
                                          dec_targets, dec_mask, idx=max_id , gsize=grid, img_size=img_size)

            ##### add noises to the negative samples
            if np.sum(enc_mask) == ngrids:
                num_rand_nodes = np.random.randint(1, 3)
                rand_dec_inputs = np.random.randint(0, ngrids, (num_rand_nodes, 2))
                dec_inputs[:len(rand_dec_inputs)] = rand_dec_inputs
                grid_xy = np.array([rand_dec_inputs[:, 0]//grid, rand_dec_inputs[:, 1]%grid])
                enc_mask[grid_xy[0, :]*grid+grid_xy[1, :]] = 0
                dec_mask[:len(rand_dec_inputs)] = 0
            
            all_enc_cat_targets.append(enc_cat_targets)
            all_enc_reg_targets.append(enc_reg_targets)
            all_enc_mask.append(enc_mask)
    #         print('len of all_enc_mask: ', len(all_enc_mask))
            all_dec_inputs.append(dec_inputs)
            all_dec_targets.append(dec_targets)
            all_dec_mask.append(dec_mask)
    
    all_enc_cat_targets = np.array(all_enc_cat_targets)
    all_enc_reg_targets = np.array(all_enc_reg_targets)
    all_enc_mask = np.array(all_enc_mask)
    all_dec_inputs = np.array(all_dec_inputs)
    all_dec_targets = np.array(all_dec_targets)
    all_dec_mask = np.array(all_dec_mask)
    rot_x_train = np.array(rot_x_train)

    return rot_x_train, all_enc_cat_targets, all_enc_reg_targets, all_enc_mask, \
        all_dec_inputs, all_dec_targets, all_dec_mask
