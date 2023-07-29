# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:41:06 2022

@author: msharma
"""

import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import tensorly as tl
tl.set_backend('pytorch')

### Fix seeds for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)


class ttd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 bias=True, dilation=1, groups=1, padding_mode='zeros', opt=None, trans=False, flag=True):
        super(ttd, self).__init__()
        self.opt = opt
        self.in_channels = in_channels//groups if not trans else in_channels
        self.out_channels = out_channels if not trans else out_channels//groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if not trans else dilation * (kernel_size - 1)
        self.dilation= dilation
        self.groups = groups
        self.trans = trans
        self.padding_mode = padding_mode
        self.bias = bias
        
        act_loss_split = self.opt.act_loss.split('-')
        if len(act_loss_split) == 1:
            self.act_loss = act_loss_split[0]
        elif len(act_loss_split) == 3:
            self.act_loss, self.red = act_loss_split[:2]
        
        if self.opt.act_loss1 != '':
            act_loss1_split = self.opt.act_loss1.split('-')
            if len(act_loss1_split) == 1:
                self.act_loss1 = act_loss1_split[0]
            elif len(act_loss1_split) == 3:
                self.act_loss1, self.red1 = act_loss1_split[:2]
        else:
            self.act_loss1, self.red1 = '', ''
        
        self.type = 'default'
        if self.opt.ttd and flag and (True if self.opt.tt_stride else self.stride==1):
            if self.kernel_size > 1 and self.groups == 1:
                self.type = 'ttd'
            elif self.kernel_size > 1 and self.groups > 1 and self.opt.tt_depthwise:
                self.type = 'ttd'
            elif self.kernel_size == 1 and self.opt.tt_pointwise and self.groups == 1:
                self.type = 'ttd'
            elif self.kernel_size == 1 and self.opt.tt_pointwise and self.groups > 1 and self.opt.tt_depthwise:
                self.type = 'ttd'
        
        if self.type == 'ttd':
            if self.padding_mode != 'zeros':
                if isinstance(self.padding, str):
                    self._reversed_padding_repeated_twice = [0, 0] * len((kernel_size, kernel_size))
                    if self.padding == 'same':
                        for d, k, i in zip((dilation, dilation), (kernel_size, kernel_size), 
                                           range(len((kernel_size, kernel_size)) - 1, -1, -1)):
                            total_padding = d * (k - 1)
                            left_pad = total_padding // 2
                            self._reversed_padding_repeated_twice[2 * i] = left_pad
                            self._reversed_padding_repeated_twice[2 * i + 1] = (total_padding - left_pad)
                else:
                    self._reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))
            
            self.factors = self.f_shape()
            self.factors0 = nn.Parameter(self.factors[0])
            self.factors1 = nn.Parameter(self.factors[1])
            if len(self.factors)//2 == 2:
                self.factors2 = nn.Parameter(self.factors[2])
                self.factors3 = nn.Parameter(self.factors[3])
            self.bias = nn.Parameter(nn.init.constant_(torch.Tensor(self.out_channels), 0.0)) if self.bias else None
            
        else:
            if not self.trans:
                self.layer = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 
                                       self.padding, bias=bias, padding_mode=self.padding_mode)
            else:
                self.layer = nn.ConvTranspose2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, 
                                                self.padding, bias=bias, padding_mode=self.padding_mode)
            if not self.bias:
                self.bias = None
        
    def forward(self, x):
        if type(x) is list or type(x) is tuple:
            x, lossy_term, lossy_term1 = x[0], x[1], x[2]
        else:
            lossy_term, lossy_term1 = 0., 0.
        if self.type == 'ttd':
            out = []
            for i in range(len(self.factors)//2):
                if self.opt.fac_parts == 'lu' and i == 1:
                    break
                if i == 0:
                    weight = tl.tt_to_tensor([self.factors0, self.factors1]).reshape(self.out_channels, self.in_channels, 
                                                                                     self.kernel_size, self.kernel_size)
                elif i == 1:
                    weight = tl.tt_to_tensor([self.factors2, self.factors3]).reshape(self.out_channels, self.in_channels, 
                                                                                     self.kernel_size, self.kernel_size)
                if not self.trans:
                    if self.padding_mode == 'zeros':
                        temp = F.conv2d(x, weight, bias=self.bias, stride=self.stride, 
                                        padding=self.padding, dilation=self.dilation, groups=self.groups)
                    else:
                        temp = F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode), 
                                        weight, bias=self.bias, stride=self.stride, 
                                        padding=self.padding, dilation=self.dilation, groups=self.groups)
                else:
                    temp = F.conv_transpose2d(x, weight.permute([1,0,2,3]), bias=self.bias, stride=self.stride, 
                                              padding=self.padding, dilation=self.dilation, groups=self.groups)
                out.append(temp)
            
            if len(out) > 1:
                fout = out[0] + out[1]
                #fout = out[0] + (F.tanh(out[0]))**2 * out[1]
                if self.act_loss == 'l1':
                    lossy_term += F.l1_loss(out[1], out[0], reduction=self.red)
                elif self.act_loss == 'l2':
                    lossy_term += F.mse_loss(out[1], out[0], reduction=self.red)
                elif self.act_loss == 'l1l2':
                    lossy_term += smooth_l1l2_loss(out[1], out[0], beta=1.0, reduction=self.red)
                elif self.act_loss == 'kld':
                    # lossy_term += F.kl_div(normalize_batch(out[1]), normalize_batch(out[0]), reduction=self.red, log_target=True)
                    lossy_term += kl_div(out[1], out[0], red=self.red)
                
                if self.act_loss1 == 'l1':
                    lossy_term1 += F.l1_loss(out[1], out[0], reduction=self.red1)
                elif self.act_loss1 == 'l2':
                    lossy_term1 += F.mse_loss(out[1], out[0], reduction=self.red1)
                elif self.act_loss1 == 'l1l2':
                    lossy_term1 += smooth_l1l2_loss(out[1], out[0], beta=1.0, reduction=self.red1)
                elif self.act_loss1 == 'kld':
                    # lossy_term1 += F.kl_div(normalize_batch(out[1]), normalize_batch(out[0]), reduction=self.red1, log_target=True)
                    lossy_term1 += kl_div(out[1], out[0], red=self.red1)
            else:
                fout = out[0]
            
            if self.act_loss == 'ortho' or self.act_loss1 == 'ortho':
                m0 = torch.concat((self.factors0, self.factors2), dim=2)[0] if len(self.factors)//2 == 2 else self.factors0[0]
                m1 = torch.concat((self.factors1, self.factors3), dim=0)[:,:,0] if len(self.factors)//2 == 2 else self.factors1[:,:,0]
                m0 = m0/torch.linalg.vector_norm(m0, dim=0, keepdim=True)
                m1 = m1/torch.linalg.vector_norm(m1, dim=1, keepdim=True)
                err0 = torch.matmul(m0.T, m0) - torch.eye(m0.shape[-1], device=m0.device)
                err1 = torch.matmul(m1, m1.T) - torch.eye(m1.shape[0], device=m1.device)
                if self.act_loss == 'ortho':
                    if self.red == 'mean':
                        lossy_term += torch.mean(err0**2) + torch.mean(err1**2)
                    elif self.red == 'sum':
                        lossy_term += torch.sum(err0**2) + torch.sum(err1**2)
                else:
                    if self.red1 == 'mean':
                        lossy_term1 += torch.mean(err0**2) + torch.mean(err1**2)
                    elif self.red1 == 'sum':
                        lossy_term1 += torch.sum(err0**2) + torch.sum(err1**2)
            return fout, lossy_term, lossy_term1
        else:
            return self.layer(x), lossy_term, lossy_term1
    
    def f_shape(self):
        if self.opt.ttd_type == 'SC-l^2':
            temp = [self.out_channels * self.in_channels, self.kernel_size**2]
        elif self.opt.ttd_type == 'S-Cl^2':
            temp = [self.out_channels, self.in_channels * self.kernel_size**2]
        
        rank = int(np.min(temp))
        options = np.arange(0.1, 0.2, 0.005) + 0.25
        factors = []
        for i, cf in enumerate(self.opt.cont_facs):
            f = max(1, int(sum(self.opt.cont_facs[:i+1]) * rank)) - (0 if i == 0 else int(sum(self.opt.cont_facs[:i]) * rank))
            if f < 1:
                break
            factors.append(torch.normal(0.0, random.choice(options), (1, temp[0], f)))
            factors.append(torch.normal(0.0, random.choice(options), (f, temp[1], 1)))
        return factors
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}' 
              ', kernel_size=({kernel_size},{kernel_size})' 
              ', stride=({stride},{stride})'
              ', padding=({padding},{padding})'
              ', padding_mode={padding_mode}'
              ', dilation={dilation}'
              ', groups={groups}'
              ', trans={trans}')
        if self.bias is None:
            s += ', bias=False'
        else:
            s += ', bias=True'
        return s.format(**self.__dict__)


def normalize_batch(x):
    shape = x.shape
    x = x.view(x.size(0), -1)
    x -= x.min(1, keepdim=True)[0]
    x_max = x.max(1, keepdim=True)[0]
    x_max[x_max == 0.] = 1e-8
    x /= x_max
    x[x == 0.] = 1e-8
    x[x != x] = 1e-8
    x = x.view(shape)
    return torch.log(x)


def kl_div(x1, x0, red='mean'):
    x0 = x0.reshape(x0.shape[0], -1)
    x1 = x1.reshape(x1.shape[0], -1)
    
    x0_mean = x0.mean(0, keepdim=True)
    x1_mean = x1.mean(0, keepdim=True)
    
    x0_std = x0.std(0, keepdim=True) + 1e-8
    x1_std = x1.std(0, keepdim=True) + 1e-8
    x1_std[x1_std < 1e-1] = 1e-1
    
    kld = torch.log(x1_std/x0_std) + (x0_std**2 + (x0_mean - x1_mean)**2)/(2 * x1_std**2) - 0.5
    if red == 'mean':
        return kld.mean(1)
    elif red == 'sum':
        return kld.sum(1)


def smooth_l1l2_loss(x, y, beta=1.0, reduction='mean'):
    loss = 0
    diff = x - y
    mask = (diff.abs() > beta)
    loss += mask * (0.5*diff**2 / beta)
    loss += (~mask) * (diff.abs() - 0.5*beta)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    

