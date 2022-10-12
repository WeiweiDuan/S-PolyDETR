from models.graph import build
import argparse
import sys, os
import torch
from util.misc import nested_tensor_from_tensor_list, NestedTensor
import numpy as np
import cv2
import copy
from data_process.gen_node_line_targets import gen_target_nodes_edges
from util.args import get_args_parser

parser = get_args_parser()
args = parser.parse_args(sys.argv[1:])
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(args.device)

model, criterion = build(args)
model.to(device)

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
model_name = args.saved_model_name

##### param: shift_range=(x_range, y_range), translate [-x_range, x_range], [-y_range, y_range]
data_dir = args.data_dir
png_map_name = args.png_map_name
png_label_name = args.png_label_name
tif_map_name = args.tif_map_name
shp_label_name = args.shp_label_name
object_list = args.object_name.split(',')
object_num = args.object_num.split(',')
object_num_int = [int(x) for x in object_num]

x_train_color, enc_cat_targets, enc_reg_targets, enc_mask,dec_inputs, dec_targets, dec_mask \
    = gen_target_nodes_edges(data_dir, png_map_name, png_label_name, tif_map_name, shp_label_name, object_list, object_num_int,\
                             grid=int(args.grid_size), img_size=args.img_size, num_dec_nodes=args.num_dec_nodes,\
                              shift_range=(args.translation_range, args.translation_range), num_shift=args.translation_num) 

enc_reg_targets = enc_reg_targets/args.grid_size
dec_inputs = dec_inputs/255.0


x_train_color = np.array(x_train_color).astype(np.float64)


x_train = x_train_color/255.0
x_train = (x_train - img_mean) / img_std

x_torch = torch.from_numpy(x_train.copy()).type(torch.FloatTensor).to(device)
x_torch = torch.moveaxis(x_torch, 3, 1)

targets = {}

enc_cat_targets_ts = torch.from_numpy(enc_cat_targets.copy()).type(torch.LongTensor).to(device)
enc_reg_targets_ts = torch.from_numpy(enc_reg_targets.copy()).type(torch.FloatTensor).to(device)
enc_mask_ts = torch.from_numpy(enc_mask.copy()).type(torch.BoolTensor).to(device)
dec_inputs_ts = torch.from_numpy(dec_inputs.copy()).type(torch.FloatTensor).to(device)
dec_targets_ts = torch.from_numpy(dec_targets.copy()).type(torch.LongTensor).to(device)
dec_mask_ts = torch.from_numpy(dec_mask.copy()).type(torch.BoolTensor).to(device)

targets['nodes_cat'] = enc_cat_targets_ts
targets['nodes_reg'] = enc_reg_targets_ts
targets['edges'] = dec_targets_ts
targets['mask'] = dec_mask_ts

print('x_torch shape: ', x_torch.shape)
print('enc_cat_targets shape: ', enc_cat_targets.shape)
print('enc_reg_targets shape: ', enc_reg_targets.shape)
print('enc_mask shape: ', enc_mask.shape)
print('dec_targets shape: ', dec_targets.shape)


enc_inputs_nested = NestedTensor(x_torch, enc_mask_ts)
dec_inputs_nested = NestedTensor(dec_inputs_ts, dec_mask_ts)

model_without_ddp = model

############################################################
##### load trained model
############################################################
if args.resume and args.trained_model_path != None:
    checkpoint = torch.load(args.trained_model_path)
    model.load_state_dict(checkpoint['model'])
    torch.cuda.empty_cache()

param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]

# optimizer = torch.optim.Adadelta(param_dicts, lr=args.lr)
optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

max_norm = args.clip_max_norm
s_epoch = args.start_epoch
epoch = args.epochs
bs = args.batch_size
prev_loss = 10000.0
num_img = x_torch.shape[0]-1
indices = np.arange(0, num_img, 1, dtype=int)

############################################################
######## model training
############################################################
for e in range(s_epoch, s_epoch+epoch):
    model.train()
    criterion.train()
    total_loss, enc_cat_loss, enc_reg_loss, dec_loss, aux_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(0, x_train.shape[0], bs):
        sampled_indices = np.random.choice(indices, size=bs, replace=False)    
        outputs, enc_attn, dec_attn, _ = \
            model(enc_inputs_nested.get_subset(sampled_indices), dec_inputs_nested.get_subset(sampled_indices))
        
        sub_target = {'cat_nodes': enc_cat_targets_ts[sampled_indices], \
                      'reg_nodes': enc_reg_targets_ts[sampled_indices], \
                      'edges': dec_targets_ts[sampled_indices], \
                      'mask': dec_mask_ts[sampled_indices]}
        loss_dict = criterion(outputs, sub_target)
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
#         lr_scheduler.step()
        losses.detach_()
        torch.cuda.empty_cache()
        total_loss += losses.item()
        enc_cat_loss += loss_dict['loss_cat_nodes'].item()
        enc_reg_loss += loss_dict['loss_reg_nodes'].item()
        dec_loss += loss_dict['loss_pos_nodes'].item()

    print('%d epoch, total loss=%f, cat loss=%f, reg loss=%f, line loss=%f'%\
              (e, total_loss, enc_cat_loss, enc_reg_loss, dec_loss))
    if total_loss < prev_loss:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
                }, '%s/%s'%(args.saved_model_dir, model_name))
        prev_loss = total_loss 
    if e % 30 == 0 and e>0:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
                }, '%s/%s'%(args.saved_model_dir, model_name[:-4]+'_e%d.pth'%(e)))

torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
                }, '%s/%s'%(args.saved_model_dir, model_name[:-4]+'_final.pth'))