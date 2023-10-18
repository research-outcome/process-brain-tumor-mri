import torch
import torch.nn as nn
from torch.nn import functional as F


class EnsembleModel(nn.Module):
    """ 
    The ensemble class for merging 4 models corresponding to 4 distinct parametric MRI folders.
    """

    def __init__(self, architecture, weights):
        super().__init__()
        flair = architecture()
        flair.load_state_dict(torch.load(weights), strict=False)
        self.flair = self.resetLinear(flair)
        t1w = architecture()
        t1w.load_state_dict(torch.load(weights), strict=False)
        self.t1w = self.resetLinear(t1w)
        t1wce = architecture()
        t1wce.load_state_dict(torch.load(weights), strict=False)
        self.t1wce = self.resetLinear(t1wce)
        t2w = architecture()
        t2w.load_state_dict(torch.load(weights), strict=False)
        self.t2w = self.resetLinear(t2w)
        self.classifier = nn.Linear(in_features=8, out_features=2)
        return
    
    def forward(self, x):
        r1 = self.flair(x[0])
        r2 = self.t1w(x[1])
        r3 = self.t1wce(x[2])
        r4 = self.t2w(x[3])
        cat = torch.cat((r1, r2, r3, r4), dim=1)
        x = self.classifier(F.relu(cat))
        return x
    
    def resetLinear(model):
        ptLayers = nn.Sequential(*list(model.children())[:-1])
 
        layers = list(ptLayers.children())
 
 
        for i in range(len(layers) - 3):
            layers[i].requires_grad_ = False

        return nn.Sequential(layers, nn.LazyLinear(2))