import cv2
import numpy as np

##### order the nodes in nodes_list
def next_position(img, cur_pos, visited): 
    cand_pos = []
    cur_x, cur_y = cur_pos
    height, width = img.shape[:2]
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if cur_x+i >= height or cur_y+j >= width:
                continue
            cand_pos = [cur_x+i, cur_y+j]
            if not visited[cur_x+i, cur_y+j] and img[cur_x+i, cur_y+j] == 255:
                visited[cur_x+i, cur_y+j] = True
                return cand_pos
    return None 

def gen_single_line_target(line_list, enc_cat_targets, enc_reg_targets, enc_mask, dec_inputs, dec_targets,\
                                  dec_mask, idx, gsize=32, img_size=256):

    line = line_list[idx]
    raster_line = np.zeros((img_size, img_size))
    for p_idx in range(len(line)-1):
        s_x, s_y = line[p_idx][0], line[p_idx][1] 
        e_x, e_y = line[p_idx+1][0], line[p_idx+1][1]
        cv2.line(raster_line, (s_y, s_x), (e_y, e_x) , 255, 1)

    start_ind = int(np.sum(enc_cat_targets))
    
    nodes_list = []
    stride = img_size // gsize
    ##### gen targets for encoder(nodes) #####
    for grid_x in range(img_size//gsize):
        for grid_y in range(img_size//gsize):
            grid = raster_line[grid_x*gsize:(grid_x+1)*gsize, \
                               grid_y*gsize:(grid_y+1)*gsize] / 255
            if not np.any(grid):
                continue
            nonzero_indices = np.where(grid!=0)
            avg_x, avg_y = np.mean(nonzero_indices[0]), np.mean(nonzero_indices[1])
            avg_x_rescale, avg_y_rescale = int(avg_x + grid_x*gsize), int(avg_y + grid_y*gsize)
            if enc_cat_targets[grid_x*stride+grid_y] == 1:
                continue

            nodes_list.append([avg_x_rescale, avg_y_rescale])
            enc_cat_targets[grid_x*stride+grid_y] = 1
            enc_reg_targets[grid_x*stride+grid_y] = [avg_x, avg_y]
    #########################
    nodes_list = np.array(nodes_list)
    if nodes_list.shape[0] == 0:
        return enc_cat_targets, enc_reg_targets, enc_mask, dec_inputs, dec_targets, dec_mask
    ##### order the nodes
    sorted_nodes = []
    for p_idx in range(len(line)-1):
        s_x, s_y = line[p_idx][0], line[p_idx][1] 
        e_x, e_y = line[p_idx+1][0], line[p_idx+1][1]
        temp_img = np.zeros((img_size, img_size)) # draw a segment
        cv2.line(temp_img, (s_y, s_x), (e_y, e_x) , 255, 1)
        visited_pos =  np.zeros((img_size, img_size)).astype('bool')
        cur_position = [s_x, s_y]
        visited_pos[s_x,s_y] = True
        while cur_position != None:
            flex_cur_pos = []
            height, width = temp_img.shape[:2]
            for i in [-4,-3,-2,-1, 0, 1,2,3,4]:
                for j in [-4,-3,-2,-1, 0, 1,2,3,4]:
                    if cur_position[0]+i >= height or cur_position[1]+j >= width:
                        continue
                    flex_cur_pos.append([cur_position[0]+i, cur_position[1]+j])
            for flex_cur in flex_cur_pos:
                if flex_cur in nodes_list.tolist() and flex_cur not in sorted_nodes:
                    sorted_nodes.append(flex_cur)
                    break
            cur_position = next_position(temp_img, cur_position, visited_pos)

    for i, node in enumerate(sorted_nodes):
        dec_inputs[start_ind+i] = np.array(node)
        dec_targets[start_ind+i] = start_ind+i
        dec_mask[start_ind+i] = 0

    enc_mask = (1 - enc_cat_targets)*enc_mask
    return enc_cat_targets, enc_reg_targets, enc_mask, dec_inputs, dec_targets, dec_mask