import torch.nn as nn
from .detection import LineVectorizer


class RoofMapNet(nn.Module):
    def __init__(self,depth,head,num_stacks,num_blocks,num_classes):

        super(RoofMapNet, self).__init__()
        self.detection = LineVectorizer(depth =depth, head=head,num_stacks = num_stacks, num_blocks = num_blocks, num_classes = num_classes)

    def forward(self, input_dict):
        results = self.detection(input_dict)
        
        return results
