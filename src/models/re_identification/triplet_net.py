"""Dummy model"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from torch import nn
import torchvision.models as models 

from src.models.base_model import BaseModel
import ipdb

class TripletNet(BaseModel):

    def __init__(self, device, base, pretrained = True):
        """Initialize the model

        Args:
            device: the device (cpu/gpu) to place the tensors
            base: (str) name of base model
            n_classes: (int) number of classes
        """
        super().__init__(device)

        self.base = base
        self.pretrained = pretrained
        self.build_model()

    def set_parameter_requires_grad(self, is_features_extracter):
        """Set requires grad for all model's parameters
        Args:
            is_feature_extracter: (bool) if we only use base model as feature extractor (no training)
        """
        if is_features_extracter:
            for param in self.model.parameters():
                param.requires_grad = False

    def build_model(self):
        """Build model architecture
        """
        if (self.base == 'resnet101'):
            self.model = models.resnet101(pretrained = self.pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            assert "Model {} is not supported! ".format(self.base)


    def forward(self, input_tensor):
        """Forward function of the model
        Args:
            input_tensor: pytorch input tensor
        """
        
        y_pred = self.model(input_tensor).squeeze()
        return y_pred
