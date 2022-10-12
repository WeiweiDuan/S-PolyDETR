import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set S-PolyDETR for prediction', add_help=False)
#     add -f to avoid the error
    parser.add_argument('-f', default=None, type=str)
    
    # Testing data
    parser.add_argument('--obj_name', default='railroads', type=str,
                       help='desired object name')
    parser.add_argument('--map_name', default='CO_Louisville_1965_degeo', type=str,
                       help='map names for prediction')
    parser.add_argument('--png_map_dir', default='/data/weiweidu/data/USGS_data/CO_Louisville_1965', type=str,
                       help='the dir to png map and label images')
    parser.add_argument('--tif_map_dir', default='/data/weiweidu/data/USGS_data/CO_Louisville_1965', type=str,
                       help='the dir to tif map and label images')
    parser.add_argument('--pred_dir', default='./pred_maps', type=str,
                       help='the dir to save predicted map')
    parser.add_argument('--img_size', default=256, type=float,
                       help='size of a training image')
    parser.add_argument('--crop_stride', default=50, type=int,
                       help='crop a image every stride')
    # Model path
    parser.add_argument('--trained_model_path', default='./trained_models/test.pth',
                       help='load the trained model for prediction')
    
    # Testing settings
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu', default='3', help='gpu name')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--buffer_size', default=8, type=int,
                       help="conflat the predicted polylines in overlapped cropped images")
    parser.add_argument('--adj_prob_thres', default=0.5, type=float,
                       help="probability threshold for existing an edge btwe two nodes")
    parser.add_argument('--vote_thres', default=3, type=int,
                       help="predict the edge if the edge is predicted more than vote_thres in several cropped images ")
    
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use") #resnet50
    parser.add_argument('--dilation', default=True, action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--masks', default=False, action='store_true',
                        help="Train segmentation head for the backbone")
    parser.add_argument('--eos_coef', default=50.0, type=float,
                        help="placeholder")

    # * PolyDETR hyperparams
    parser.add_argument('--grid_size', default=16, type=int,
                       help="size of grid, one grid cell in a cropped image proposes one candidate node")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks") #2048
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)") #256
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer") #was 0.1
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', default=True, action='store_true')
    parser.add_argument('--num_dec_nodes', default=150, type=int,
                       help='max num of nodes for the decoder input')

    return parser
