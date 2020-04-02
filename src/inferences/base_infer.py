"""Image Base dataset"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as osp
from PIL import Image

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import pandas as pd

class BaseInfer():
    """Defines further steps afer getting outputs from model, e.g. 
       calculate additional metrics: accuracy, plot confusion mat, ...
       export prediction results to txt file
    """
    def __init__(self):
        pass

    def init_metric(self):
        pass

    def batch_update(self, outputs, labels):
        pass
        """
            This function is called every batch
        """
        

    def finalize_metric(self, logger, test_loss):
        """
        This function is called at the end of evaluation process
        Final statistic results are given
        """
        return test_loss
