import torch
import cv2
import sys, os
import numpy as np
from util.args_test import get_args_parser

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(args.device)

def predict(model, x_torch_sub):
    with torch.no_grad():
        outputs, enc_attn, dec_attn, _ = model.predict(x_torch_sub)
        
        node_cat_pred = torch.nn.functional.softmax(outputs['pred_cat_nodes'], dim=-1)
        node_cat_pred_np = node_cat_pred.cpu().detach().numpy()

        node_reg_pred = torch.nn.functional.sigmoid(outputs['pred_reg_nodes'])
        node_reg_pred_np = node_reg_pred.cpu().detach().numpy()

        node_pos_pred = torch.nn.functional.softmax(outputs['pred_edges'], dim=-1)
        node_pos_pred_np = node_pos_pred.cpu().detach().numpy()

        pred_dec_mask_np = outputs['mask'].cpu().detach().numpy()

        pred_dec_inputs_np = outputs['dec_inputs'].cpu().detach().numpy()
        
        enc_attn_np = enc_attn.cpu().detach().numpy()
        dec_attn_np = dec_attn.cpu().detach().numpy()
        
    return node_cat_pred_np, node_reg_pred_np, node_pos_pred_np, pred_dec_inputs_np, \
            pred_dec_mask_np, enc_attn_np, dec_attn_np

def predict_raster_res(png_map_path, model, output_dir, pred_name, win_size, batch, stride):
    map_img = cv2.imread(png_map_path)
    pred_edge_map = np.zeros(map_img.shape[:2])
    pred_node_map = np.zeros(map_img.shape[:2])

    ng_in_row = int(args.img_size // args.grid_size) # num of grids in a row
    
    height, width = map_img.shape[:2]
    x_test_name = []
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            x_test_name.append(str(i)+'_'+str(j))

    for i in range(0, len(x_test_name), batch):
        sub_x_name = x_test_name[i:i+batch]
        x_test = []
        x_name_list = []
        for name in sub_x_name:
            row, col = name.split('_')
            row, col = int(row), int(col)
            img = map_img[row:row+win_size, col:col+win_size]
            if img.shape != (256,256,3):
                del name
                continue
            x_test.append(img)
            x_name_list.append(name)

        if len(x_test) == 0:
            continue

        x_test_np = np.array(x_test)
        x_test_np = x_test_np.astype(np.float64)
        x_test_np  = x_test_np /255.0
        x_test_np  = (x_test_np  - img_mean) / img_std

        x_torch = torch.from_numpy(x_test_np.copy()).type(torch.FloatTensor).to(device)
        x_torch = torch.moveaxis(x_torch, 3, 1)


        node_cat_pred_np, node_reg_pred_np, node_pos_pred_np, pred_dec_inputs_np, pred_dec_mask_np,  \
           enc_attn_np, dec_attn_np = predict(model, x_torch)

        for img_idx in range(node_cat_pred_np.shape[0]):
            ####### draw predicted nodes
            node_img = np.zeros_like(x_test_np[img_idx, :, :, 0])
            single_node_cat_pred = np.argmax(node_cat_pred_np[img_idx,:,:], axis=-1)
            pos_idx = np.where(single_node_cat_pred!=0)[0]

            if pos_idx.shape[0] < 5:
                continue

            for pts in pos_idx:
                row, col = pts//ng_in_row, pts%ng_in_row
                pp_x, pp_y = (node_reg_pred_np[img_idx][pts]*args.grid_size).astype('int32')
                pp_x, pp_y = int(pp_x+row*args.grid_size), int(pp_y+col*args.grid_size)
                cv2.circle(node_img, (pp_y, pp_x), 3, 1, -1) 

            ####### draw predicted edges
            single_node_pos = np.argmax(node_pos_pred_np[img_idx], axis=-1)
            max_num_nodes = node_pos_pred_np.shape[1]
            line = np.ones((max_num_nodes, 2))*-1

            for i, node in enumerate(pred_dec_inputs_np[img_idx]):
                if sum(node) == 0:
                    continue
                x, y = (node*255).astype('int32')
                line[int(single_node_pos[i])] = [x, y]
            line = line.astype('int32')

            edge_img = np.zeros_like(x_test_np[img_idx, :, :, 0])
            p = 0
            prev_node = line[p]
            while True:
                if p >= max_num_nodes-1:
                    break
                if line[p].sum() == -2 and p == 0:
                    prev_node = line[p+1]
                    p += 1
                    continue
                elif line[p+1].sum() == -2:
                    p += 1
                    continue
                x_s, y_s = prev_node
                x_e, y_e = line[p+1]
                if x_s == x_e and y_s == y_e:
            #         print('same')
                    p += 1
                    continue
                if abs(x_s-x_e) > 25 or abs(y_s-y_e) > 25:
                    prev_node = line[p+1]
                    p += 1
                    continue
                cv2.line(edge_img, (y_s, x_s), (y_e, x_e), 1, 1) #
                prev_node = np.array([x_e, y_e])
                p += 1

            row, col = x_name_list[img_idx][:-4].split('_')
            row, col = int(row), int(col)

            if not np.any(node_img):
                continue

            pred_edge_map[row:row+256, col:col+256] = np.logical_or(edge_img, pred_edge_map[row:row+256, col:col+256])

    ##### dilate the drawn polylines to conflate very close polylines into one
    kernel = np.ones((3,3), np.uint8)
    pred_dilate = cv2.dilate(pred_edge_map*255, kernel, iterations=2)

    ##### thining the results into 1-pixel width
    pred_thin = np.zeros_like(pred_dilate)
    pred_thin = cv2.ximgproc.thinning(pred_dilate, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    
    cv2.imwrite(os.path.join(output_dir, pred_name+'.png'), pred_thin)
    print('save the predicted map in %s'%(os.path.join(output_dir, pred_name+'.png')))
