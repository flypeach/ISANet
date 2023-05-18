import mindspore as ms
import math
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

import torch
import math
from torch.nn import functional as F
import torch.nn as torch_nn
import numpy as np

class SelfAttentionBlock2D_ms(nn.Cell): # 迁移完成
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, bn_type=None):
        super(SelfAttentionBlock2D_ms, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.SequentialCell([
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, has_bias=False, pad_mode='valid'),
            nn.BatchNorm2d(self.key_channels, use_batch_statistics=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, has_bias=False, pad_mode='valid'),
            nn.BatchNorm2d(self.key_channels, use_batch_statistics=True),
            nn.ReLU()
            ]
        )
        self.f_query = nn.SequentialCell([
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, has_bias=False, pad_mode='valid'),
            nn.BatchNorm2d(self.key_channels, use_batch_statistics=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, has_bias=False, pad_mode='valid'),
            nn.BatchNorm2d(self.key_channels, use_batch_statistics=True),
            nn.ReLU()]
        )

        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1, has_bias=False, pad_mode='valid')
        self.W = nn.SequentialCell(
            nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels, kernel_size=1, has_bias=False, pad_mode='valid'),
            nn.BatchNorm2d(self.out_channels, use_batch_statistics=True),
            nn.ReLU(),
        )
                  

    def construct(self, x):
        batch_size, h, w = x.shape[0], x.shape[2], x.shape[3]

        value = self.f_value(x).view((batch_size, self.value_channels, -1))
        value = ops.transpose(value, (0, 2, 1))
        query = self.f_query(x).view((batch_size, self.key_channels, -1))
        query = ops.transpose(query, (0, 2, 1))
        key = self.f_key(x).view((batch_size, self.key_channels, -1))

        # matmul = ops.BatchMatMul()
        sim_map = ops.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        softmax = ops.Softmax(axis=-1) # -1
        sim_map = softmax(sim_map)

        context = ops.matmul(sim_map, value)
        # context = ops.Reshape(ops.transpose(context, (0, 2, 1)), (batch_size, self.value_channels, h, w)) # torch.reshape() 大致相当于tensor.contiguous().view()
        context = ops.transpose(context, (0, 2, 1))
        context = context.view((batch_size, self.value_channels, h, w))
        context = self.W(context)
        return context
    
    
class ISA_Block_ms(nn.Cell): # 迁移完成
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factor=[8,8], bn_type=None):
        super(ISA_Block_ms, self).__init__()
        self.out_channels = out_channels
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.down_factor = down_factor
        self.long_range_sa = SelfAttentionBlock2D_ms(in_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
        self.short_range_sa = SelfAttentionBlock2D_ms(out_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
    
    def construct(self, x):
        n, c, h, w = x.shape
        dh, dw = self.down_factor       # down_factor for h and w, respectively
        
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        # pad the feature if the size is not divisible
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:  # padding in both left&right sides
            pad_op = nn.Pad(paddings=((0, 0), (0, 0), (pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)))
            # pad_op = nn.Pad(paddings=((0, 0), (0, 0), (pad_h - pad_h//2, pad_h//2), (pad_w - pad_w//2, pad_w//2)))
            feats = pad_op(x)
        else:
            feats = x
        
        # long range attention
        feats = feats.view((n, c, out_h, dh, out_w, dw))
        #feats = feats.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)
        feats = ops.transpose(feats, (0, 3, 5, 1, 2, 4))
        feats = feats.view((-1, c, out_h, out_w))
        feats = self.long_range_sa(feats)
        c = self.out_channels

        # short range attention
        feats = feats.view((n, dh, dw, c, out_h, out_w))
        #feats = feats.permute(0, 4, 5, 3, 1, 2).contiguous().view(-1, c, dh, dw)
        feats = ops.transpose(feats, (0, 4, 5, 3, 1, 2))
        feats = feats.view((-1, c, dh, dw))
        feats = self.short_range_sa(feats)
        #feats = feats.view(n, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5)
        feats = feats.view((n, out_h, out_w, c, dh, dw))
        feats = ops.transpose(feats, (0, 3, 1, 4, 2, 5))
        #feats = feats.contiguous().view(n, c, dh * out_h, dw * out_w)
        feats = feats.view((n, c, dh * out_h, dw * out_w))

        # remove padding
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w]
        
        return feats    
    
    
    
class SelfAttentionBlock2D_py(torch_nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, bn_type=None):
        super(SelfAttentionBlock2D_py, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = torch_nn.Sequential(
            torch_nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False),
            torch_nn.BatchNorm2d(self.key_channels),
            torch_nn.ReLU(),
            torch_nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, bias=False),
            torch_nn.BatchNorm2d(self.key_channels),
            torch_nn.ReLU(),
        )
        self.f_query = torch_nn.Sequential(
            torch_nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False),
            torch_nn.BatchNorm2d(self.key_channels),
            torch_nn.ReLU(),
            torch_nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, bias=False),
            torch_nn.BatchNorm2d(self.key_channels),
            torch_nn.ReLU(),
        )

        self.f_value = torch_nn.Conv2d(self.in_channels, self.value_channels, kernel_size=1, bias=False)
        self.W = torch_nn.Sequential(
            torch_nn.Conv2d(self.value_channels, self.out_channels, kernel_size=1, bias=False),
            torch_nn.BatchNorm2d(self.out_channels),
            torch_nn.ReLU(),
        )
                  

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous() ###10.10 迁移到这里了
        context = context.view(batch_size, self.value_channels, h, w)
        context = self.W(context)
        return context

    
class ISA_Block_py(torch_nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, down_factor=[8,8], bn_type=None):
        super(ISA_Block_py, self).__init__()
        self.out_channels = out_channels
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.down_factor = down_factor
        self.long_range_sa = SelfAttentionBlock2D_py(in_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
        self.short_range_sa = SelfAttentionBlock2D_py(out_channels, key_channels, value_channels, out_channels, bn_type=bn_type)
    
    def forward(self, x):
        n, c, h, w = x.size()
        dh, dw = self.down_factor       # down_factor for h and w, respectively
        
        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        # pad the feature if the size is not divisible
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:  # padding in both left&right sides
            feats = F.pad(x, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
        else:
            feats = x
        
        # long range attention
        feats = feats.view(n, c, out_h, dh, out_w, dw)
        feats = feats.permute(0, 3, 5, 1, 2, 4).contiguous().view(-1, c, out_h, out_w)
        feats = self.long_range_sa(feats)
        c = self.out_channels

        # short range attention
        feats = feats.view(n, dh, dw, c, out_h, out_w)
        feats = feats.permute(0, 4, 5, 3, 1, 2).contiguous().view(-1, c, dh, dw)
        feats = self.short_range_sa(feats)
        feats = feats.view(n, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5)
        feats = feats.contiguous().view(n, c, dh * out_h, dw * out_w)

        # remove padding
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2:pad_h//2 + h, pad_w//2:pad_w//2 + w]
        
        return feats    
    
    
if __name__ == "__main__":
    
    x = np.random.uniform(-1, 1, (1, 1024, 56, 56)).astype(np.float32)
    # py_net = SelfAttentionBlock2D_py(in_channels=1024, key_channels=256, value_channels=512, out_channels=512)
    # ms_net = SelfAttentionBlock2D_ms(in_channels=1024, key_channels=256, value_channels=512, out_channels=512)
    
    
    py_net = ISA_Block_py(in_channels=1024, key_channels=256, value_channels=512, out_channels=512)
    ms_net = ISA_Block_ms(in_channels=1024, key_channels=256, value_channels=512, out_channels=512)
    
    for m in py_net.modules():
        if isinstance(m, torch_nn.Conv2d):
            torch_nn.init.constant_(m.weight, 0.1)
            
    for _, cell in ms_net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(ms.common.initializer.initializer(0.1, cell.weight.shape, cell.weight.dtype))
            
            
    y_ms = ms_net(ms.Tensor(x))
    y_pt = py_net(torch.from_numpy(x))
    diff = np.max(np.abs(y_ms.asnumpy() - y_pt.detach().numpy()))
    print(diff)