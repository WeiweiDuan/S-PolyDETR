from models.graph import build
import argparse
import sys, os
import torch
import numpy as np
import cv2
import copy
from util.args_test import get_args_parser
import networkx as nx
from test_helper import predict_raster_res
from util.args_test import get_args_parser
from util.util_helper import remove_files, create_folder

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(args.device)

model_path = args.trained_model_path
output_dir = args.pred_dir
DATA_DIR = args.png_map_dir
MAP_PATH = os.path.join(DATA_DIR, args.map_name+'.png')
pred_name = args.map_name+'_'+args.obj_name+'_pred'

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])
args.dropout = 0.0
device = torch.device(args.device)

model, criterion = build(args)
model.to(device)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
torch.cuda.empty_cache()

batch = args.batch_size
win_size = args.img_size
stride = args.crop_stride


if not os.path.exists(MAP_PATH):
    tif_map_path = os.path.join(args.tif_map_dir, MAP_NAME[:-4]+'.tif')
    tif2png(tif_map_path, MAP_PATH)


predict_raster_res(MAP_PATH, model, output_dir, pred_name, win_size, batch, stride)