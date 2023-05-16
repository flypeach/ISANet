import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from isablock import ISA_Module
from backbone import ResNet



class ISANet(nn.Cell): # 迁移完成
    def __init__(self, num_classes):
        super(ISANet, self).__init__()
        self.ISAHead = ISA_Module(in_channels=2048, key_channels=256, value_channels=512, out_channels=512, dropout=0)
        self.backbone = ResNet.resnet50(replace_stride_with_dilation=[1,2,4])
        self.Conv_1 = nn.SequentialCell(
            nn.Conv2d(512, 512, 3, pad_mode = 'pad', padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) 
        self.Upsample_1 = nn.ResizeBilinear() # nn.ResizeBilinear(scale_factor=8, mode="bilinear", align_corners=True)
        self.cls_seg = nn.Conv2d(512, num_classes, 3, pad_mode = 'pad', padding=1)

    def forward(self, x):
        """Forward function."""
        output = self.backbone(x)
        output = self.ISAHead(output)
        output = self.Conv_1(output)
        output = self.Upsample_1(output, size=None, scale_factor=8, align_corners=True)
        output = self.cls_seg(output)
        return output

if __name__ == "__main__":
    x = ops.standard_normal((1, 1024, 56, 56))
    model = ISANet(num_classes=2)
    # print(model)
    out = model(x)
    print("GCNet output.shape:", out.shape)