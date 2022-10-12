import numpy as np
import cv2
import math

def raster_a_line(img, line):
    for p in range(len(line)-1):
        cv2.line(img, (line[p][1], line[p][0]), (line[p+1][1], line[p+1][0]), 1, 1)
    return img

def rm_dup_lines(all_lines):
    new_all_lines = []
    for i in range(len(all_lines)):
        dup_flag = False
        raster_i = np.zeros((256, 256))
        raster_i = raster_a_line(raster_i, all_lines[i])
        for j in range(len(all_lines)):
            if i == j:
                continue
            raster_j = np.zeros((256, 256))
            raster_j = raster_a_line(raster_j, all_lines[j])
            intersection = np.sum(raster_j * raster_i)
            sum_i = np.sum(raster_i)*1.0
            sum_j = np.sum(raster_j)*1.0
#             print(intersection, sum_i, sum_j)
            if intersection / sum_i > 0.8 and sum_j > sum_i:
                dup_flag = True
                break
            elif intersection / sum_i > 0.8 and sum_j == sum_i and i > j:
                dup_flag = True
                break
        if not dup_flag:
            new_all_lines.append(all_lines[i])
    return new_all_lines

##### return a list of lines, shape is (#lines, #nodes_in_line, 2)
##### one list is a line
##### a list is a list of [xs, ys]
def gen_line_list(polylines_interp, xmin, ymin, xmax, ymax):
    lines = []
    prev_node = []
    init_flag = 0
    diff_thres = 5
    line_counter = 0

    for line in polylines_interp:
        if line == []:
            continue
        for p in range(len(line)-1):
            x_s, y_s = line[p]
            x_e, y_e = line[p+1]
            if min(x_s, x_e) <= xmin or min(y_s, y_e) <= ymin \
                    or max(x_e, x_s) >= xmax or max(y_e, y_s) >= ymax:                
                continue
#             print(line)
            if init_flag == 0:
                prev_node = [x_e, y_e]
                lines.append([[x_s-xmin, y_s-ymin], [x_e-xmin, y_e-ymin]])
#                 lines.append([[x_s, y_s], [x_e, y_e]])
                init_flag = 1
            else:
                if prev_node == [x_s, y_s] or (abs(prev_node[0]-x_s)<diff_thres and abs(prev_node[1]-y_s)<diff_thres):
                    lines[line_counter].append([x_e-xmin, y_e-ymin])
#                     lines[line_counter].append([x_e, y_e])
                    prev_node = [x_e, y_e]
                else:
                    init_flag = 0
                    line_counter += 1
                    
    #### remove duplicate lines
    if len(lines) <= 1:
        return lines
    else:
        return rm_dup_lines(lines)