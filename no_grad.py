import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import hflip
from torch import nn
import os
from PIL import Image

from unimatch.unimatch.unimatch import UniMatch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

feature_channels = 128
num_scales = 1
upsample_factor = 8
num_head = 1
ffn_dim_expansion = 4
num_transformer_layers = 6
task = 'stereo'
attn_type = "self_swin2d_cross_1d"
attn_splits_list = [1]
corr_radius_list = [-1]
prop_radius_list = [-1]
num_reg_refine = 1

model = UniMatch(feature_channels=feature_channels,
                    num_scales=num_scales,
                    upsample_factor=upsample_factor,
                    num_head=num_head,
                    ffn_dim_expansion=ffn_dim_expansion,
                    num_transformer_layers=num_transformer_layers,
                    attn_type=attn_type,
                    attn_splits_list=attn_splits_list,
                    corr_radius_list=corr_radius_list,
                    prop_radius_list=prop_radius_list,
                    num_reg_refine=num_reg_refine,
                    task=task).to(device)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transform_list = [transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]

val_transform = transforms.Compose(val_transform_list)

directory = os.listdir('../data/training/image_2')

activation = {}



def get_activation(name):
    def hook(model, name, output):
        activation['backbone'] = output#torch.cat((output[0].detach(), output[1].detach()),1)
    return hook

for i in range(len(directory)):

    # if i >0:
    #     break
    # print(directory[i])
    left = np.array(Image.open('../data/training/image_2/'+ directory[i]).convert('RGB')).astype(np.float32)
    right = np.array(Image.open('../data/training/image_3/'+directory[i]).convert('RGB')).astype(np.float32)

    sample = {'left': left, 'right': right}
    sample = val_transform(sample)
    
    left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
    right = sample['right'].to(device).unsqueeze(0)
    # print(left.shape)
    activation = {}
    with torch.no_grad():
        pred_disp = model(left, right)['flow_preds']
        
    model.transformer.layers[5].register_forward_hook(get_activation('backbone'))
    # model.backbone.register_forward_hook(get_activation('backbone'))
    # model.upsampler.register_forward_hook(get_activation('backbone'))
    backbone = activation['backbone']
    backbone = backbone.squeeze(0).cpu().numpy()
    print(backbone.shape)

    string = '../data/training/backbone/'+directory[i]
    np.save(string[:-4],backbone)
