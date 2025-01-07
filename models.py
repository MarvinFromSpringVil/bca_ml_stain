from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os
from glob import glob
from unet_utils import *

class Unet(nn.Module):
    def __init__(self, img_ch, fch_base=16, isBN=True, isDeconv=True):
        super(Unet, self).__init__()
        
        self.blocks = nn.ModuleList()
        
        self.down1 = ConvNoPool(img_ch, fch_base, isBN)
        self.down2 = ConvPool(fch_base, fch_base * 2, isBN)
        self.down3 = ConvPool(fch_base * 2, fch_base * 4, isBN)
        self.down4 = ConvPool(fch_base * 4, fch_base * 8, isBN)
        
        self.encoder = ConvPool(fch_base * 8+1, fch_base * 16, isBN)
        
        self.up1 = UpsampleConv(fch_base * 16, fch_base * 8, isDeconv, isBN)
        self.up2 = UpsampleConv(fch_base * 8, fch_base * 4, isDeconv, isBN)
        self.up3 = UpsampleConv(fch_base * 4, fch_base * 2 , isDeconv, isBN)
        self.up4 = UpsampleConv(fch_base * 2, fch_base, isDeconv, isBN)
        
        self.out = ConvOut(fch_base)
        
        self.blocks = nn.ModuleList([self.down1, self.down2, self.down3,\
                                     self.down4, self.encoder, self.up1, self.up2,\
                                     self.up3, self.up4, self.out])

    def forward(self, input_, condition):
        d1 = self.down1(input_)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        #print(d4.shape) # 1 128 28 28 
        d5 = torch.cat([d4, condition], dim=1)
        #enc = self.encoder(d4)
        enc = self.encoder(d5)
        
        u1 = self.up1(enc, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        output_ = self.out(u4)
        return output_
    

def get_model(ckpt=None):
    model = Unet(img_ch=3) 

    if ckpt: 
        model.load_state_dict(torch.load(ckpt))

    return model 
