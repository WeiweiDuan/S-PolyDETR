import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer_encoder import build_transformer as build_encoder
from .transformer_decoder import build_transformer as build_decoder
from .position_encoding import PositionEmbeddingSine_ as PositionEmbeddingSine

class graph(nn.Module):
    def __init__(self, backbone, encoder, decoder, num_classes=1, aux_loss=False, grid=32, img_size=256, num_dec_nodes=32):
        super().__init__()
        self.grid_size = grid
        self.img_size = img_size
        self.num_dec_nodes = num_dec_nodes
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = encoder.d_model
        ##### map banckbone's output to low-dim
        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.backbone = backbone
        ##### map 2d coordinate to high-dim
        self.linear = nn.Linear(2, self.hidden_dim)
        ############################
        ##### encoder classifiers
        ##### classify grid into 0/1
        ############################
        self.node_classifier = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        ##### regression the precise location of node in each grid
        self.node_regression = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        ###########################
        ##### decoder nodes position classifier
        ############################
        self.edge_classifier = MLP(self.hidden_dim, self.hidden_dim, num_classes+1, 3)
        self.aux_loss = aux_loss
   
    def forward(self, enc_inputs: NestedTensor, dec_inputs: NestedTensor):
        backbone_in = enc_inputs.tensors
        if isinstance(backbone_in, (list, torch.Tensor)):
            backbone_in = nested_tensor_from_tensor_list(backbone_in)
        features, pos = self.backbone(backbone_in)
        src, mask = features[-1].decompose()

        assert mask is not None
        ##### encoder
        memory, enc_attn = self.encoder(self.input_proj(src), mask, pos[-1])
        enc_cat_outputs = self.node_classifier(memory)
        enc_reg_outputs = self.node_regression(memory)
        ##### decoder
        edge_inputs = dec_inputs.tensors
        dec_mask = dec_inputs.mask
        enc_mask = enc_inputs.mask
        
        dec_input_embed = self.linear(edge_inputs)
        dec_hs, dec_attn, enc_dec_attn = self.decoder(dec_input_embed, memory, dec_mask, enc_mask, pos[-1])
        dec_outputs = self.edge_classifier(dec_hs)
        out = {'pred_cat_nodes': enc_cat_outputs, 'pred_reg_nodes': enc_reg_outputs,\
              'pred_edges': dec_outputs, 'mask': dec_mask}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(dec_attn)
        return out, enc_attn, dec_attn, enc_dec_attn
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs):
        # as a dict having both a Tensor and a list.
        return [{'pred_edges': i} for i in outputs]
    
    def predict(self, enc_inputs):
        import numpy as np
        if isinstance(enc_inputs, (list, torch.Tensor)):
            backbone_in = nested_tensor_from_tensor_list(enc_inputs)
        features, pos = self.backbone(backbone_in)
        src, mask = features[-1].decompose()

        assert mask is not None
        
        ###### encoder
        memory, enc_attn = self.encoder(self.input_proj(src), mask, pos[-1])
        enc_cat_outputs = self.node_classifier(memory)
        enc_reg_outputs = self.node_regression(memory)
        enc_cat_sm = torch.nn.functional.softmax(enc_cat_outputs, dim=-1)
        enc_cat_bin = (enc_cat_sm > 0.5)[:,:,1].type(torch.ByteTensor)
        enc_reg_outputs_sig = torch.nn.functional.sigmoid(enc_reg_outputs)

        ##### decoder
        bs = memory.shape[0]      
        dec_inputs = torch.zeros((bs, self.num_dec_nodes, 2)).to(memory.device)
        dec_mask = torch.ones((bs, self.num_dec_nodes)).type(torch.BoolTensor).to(memory.device)
        num_grids = int((self.img_size//self.grid_size)**2)
        ng_in_row = int(self.img_size//self.grid_size)
        for i in range(bs):
            c = 0
            for j in range(num_grids):
                if enc_cat_bin[i,j] == 1:
                    grid_x, grid_y = j//ng_in_row, j%ng_in_row
                    x, y = enc_reg_outputs_sig[i, j, 0]*self.grid_size + self.grid_size*grid_x, \
                            enc_reg_outputs_sig[i, j, 1]*self.grid_size + self.grid_size*grid_y
                    dec_inputs[i, c] = torch.FloatTensor([x, y])
                    dec_mask[i, c] = 0
                    c += 1
                if c >= self.num_dec_nodes:
                    break
        
        dec_inputs = dec_inputs/(self.img_size-1)
        
        dec_inputs_embed = self.linear(dec_inputs)
        enc_mask = (1 - enc_cat_bin).type(torch.BoolTensor).to(memory.device)
        dec_hs, dec_attn, enc_dec_attn = self.decoder(dec_inputs_embed, memory, dec_mask, enc_mask, pos[-1])
        dec_outputs = self.edge_classifier(dec_hs)
        
        out = {'pred_cat_nodes': enc_cat_outputs, 'pred_reg_nodes': enc_reg_outputs,\
              'pred_edges': dec_outputs, 'mask': dec_mask, 'dec_inputs': dec_inputs}
        return out, enc_attn, dec_attn, enc_dec_attn                                   
        

class SetCriterion(nn.Module):
    
    def __init__(self, num_classes, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(2)
        empty_weight[-1] = 5.0
        self.register_buffer('empty_weight', empty_weight)
    
    def loss_cat_nodes(self, outputs, targets):
        node_logits = outputs['pred_cat_nodes']       
        loss_ce = F.cross_entropy(node_logits.transpose(1, 2), targets['cat_nodes'])
        losses = {'loss_cat_nodes': loss_ce}
        return losses
    
    def loss_reg_nodes(self, outputs, targets):
        node_reg = outputs['pred_reg_nodes'].sigmoid()
        loss_mse = F.mse_loss(node_reg, targets['reg_nodes'])
        reg_loss_mask = targets['cat_nodes']
        loss_ = torch.mean(loss_mse*reg_loss_mask)*self.eos_coef
        losses = {'loss_reg_nodes': loss_}
        return losses
    
    def loss_edges(self, outputs, targets):
        edge_logits = outputs['pred_edges']
        edge_mask_bin = ~outputs['mask']
        edge_mask_bin = edge_mask_bin.unsqueeze(2).repeat(1, 1, edge_logits.shape[-1])
        loss = F.cross_entropy(edge_logits.transpose(1, 2), targets['edges'])
        loss_ = torch.mean(loss*edge_mask_bin)
        losses = {'loss_pos_nodes': loss_}
        return losses
    
    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            'loss_cat_nodes': self.loss_cat_nodes,
            'loss_reg_nodes': self.loss_reg_nodes,
            'loss_pos_nodes': self.loss_edges
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)
    
    
    def forward(self, outputs, targets):
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))
        
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in ['loss_conn']:
                    l_dict = self.get_loss(loss, aux_outputs, targets)#, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def build(args):
    num_classes = args.num_dec_nodes-1
    device = torch.device(args.device)
    
    backbone = build_backbone(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)

    model = graph(
        backbone,
        encoder,
        decoder,
        num_classes, 
        grid=args.grid_size, 
        img_size=args.img_size, 
        num_dec_nodes=args.num_dec_nodes
    )
    
    losses = ['loss_cat_nodes', 'loss_reg_nodes', 'loss_pos_nodes']
    criterion = SetCriterion(num_classes, 
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    return model, criterion