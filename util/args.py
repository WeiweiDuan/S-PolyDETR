import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
#     add -f to avoid the error
    parser.add_argument('-f', default=None, type=str)
    
    # training settingss
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr_backbone', default=5e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=180, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu', default='1', help='gpu name')
    parser.add_argument('--resume', default='False', help='resume from checkpoint')
    parser.add_argument('--trained_model_path', default=None,
                       help='load the pre-trained model from the path')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--saved_model_dir', default='./trained_models',
                        help='path where to save, empty for no saving')
    parser.add_argument('--saved_model_name', default='test.pth',
                       help='the model name for the training')
    
    # dataset parameters
    parser.add_argument('--data_dir', default='/data/weiweidu/data/USGS_data/CO_Louisville_1965', type=str,
                       help="dir for tif/png map, png/shp labels")
    parser.add_argument('--png_map_name', default='CO_Louisville_1965_degeo.png', type=str)
    parser.add_argument('--tif_map_name', default='CO_Louisville_450543_1965_24000_geo.tif', type=str)
    parser.add_argument('--png_label_name', default='railroad_GT_Louisville.png', type=str)
    parser.add_argument('--shp_label_name', default='Louisville_railroads_perfect_1965_4269.shp', type=str)
    parser.add_argument('--object_name', default='railroads,roads', type=str, 
                       help="crop training image including the objects, separated by comma")
    parser.add_argument('--object_num', default='2,3,3', type=str,
                       help="num of images in each object category, the last num is for random negative samples")
    # Inputs
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--grid_size', type=float, default=16.0)
    parser.add_argument('--num_dec_nodes', type=int, default=150, 
                        help='max num of nodes for the decoder input')
    parser.add_argument('--translation_range', type=int, default=100, 
                        help='translation augmentation, range in [-v, v] in both x- and y-axis')
    parser.add_argument('--translation_num', type=int, default=3,
                       help='num of randomly selecting in the tranlation range')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use") #resnet50
    parser.add_argument('--dilation', default=True, action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * S-PolyDETR 
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
    
    # * Segmentation
    parser.add_argument('--masks', default=False, action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', default=False, dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Loss coefficients
    parser.add_argument('--eos_coef', default=50.0, type=float,
                        help="Relative classification weight of the object class")

    return parser
