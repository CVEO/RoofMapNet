from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from roofmapnet.config import M
from roofmapnet.models.hourglass_pose import hg



class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(M.head_size, []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(M.head_size, []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)


class MultitaskLearner(nn.Module):
    def __init__(self, depth, head,num_stacks, num_blocks, num_classes):
        super(MultitaskLearner, self).__init__()
        self.backbone = hg(depth =depth, head=head,num_stacks = num_stacks, num_blocks = num_blocks, num_classes = num_classes)
        head_size = M.head_size
        self.num_class = sum(sum(head_size, []))
        self.head_off = np.cumsum([sum(h) for h in head_size])
        self.conv_jmap = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=2,kernel_size=1)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.latlayer = nn.Conv2d(in_channels=2,out_channels=2,kernel_size=3,stride=1,padding=1)
        self.gaussalpha = nn.Parameter(torch.tensor([-0.5],dtype=torch.float32))
        self.gaussbeta = nn.Parameter(torch.tensor([2],dtype=torch.float32))

    def forward(self, input_dict):
        image = input_dict["image"]
        outputs, feature = self.backbone(image)
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape
        T = input_dict["target"].copy()
        n_jtyp = T["jmap"].shape[1]
        T_gaussjam = []
        for i in range(batch):
            jmap_array = torch.tensor(T["jmap"][i][0],device='cuda')
            coordinates = torch.argwhere(jmap_array == 1)
            x = torch.arange(128,device='cuda')
            y = torch.arange(128,device='cuda')
            X, Y = torch.meshgrid(x, y)
            for center_x, center_y in coordinates:
                distance = torch.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                jmap_array += torch.exp(self.gaussalpha * (distance / self.gaussbeta) ** 2)
            jmap_array[jmap_array <= 0.05] = 0
            jmap_array[jmap_array> 1] = 1
            new_items =jmap_array.unsqueeze(0).unsqueeze(0)
            
            T_gaussjam.append(new_items)
        T["gaussjmap"] = torch.cat(T_gaussjam, dim=0)
        
        # switch to CNHW
        for task in ["jmap"]:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["gaussjmap"]:
            T[task] = T[task].permute(1, 0, 2, 3)
        for task in ["joff"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)

        offset = self.head_off
        # loss_weight = M.loss_weight
        # losses = []
        for stack, output in enumerate(outputs):
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            gaussjmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)
            jmap_origin = self.conv_jmap(gaussjmap[0].permute(1,0,2,3))
            jmap_ds = self.maxpool(gaussjmap[0].permute(1,0,2,3))
            jmap_ds = self.latlayer(jmap_ds)
            jmap = jmap_origin + F.interpolate(jmap_ds, size=jmap_origin.shape[-2:], mode="bilinear", align_corners=True)
            jmap = jmap.permute(1,0,2,3).unsqueeze(0)
            lmap = output[offset[0] : offset[1]].squeeze(0)
            joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)
            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],
                    "lmap": lmap.sigmoid(),
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,
                }
                if input_dict["mode"] == "testing":
                    return result

        return result