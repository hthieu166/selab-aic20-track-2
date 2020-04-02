"""Factory pattern for different models and datasets"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import ipdb
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'src')))

""" > Import your datasets here """
from src.datasets.aic20_vehicle_type import AIC20_VEHI_TYPE
from src.datasets.aic20_vehicle_reid import AIC20_VEHI_REID

""" > Import your models here """
from src.models.dummy_model import DummyModel
from src.models.classifiers.img_classifier import ImageClassifier    
from src.models.re_identification.triplet_net import TripletNet

""" > Import your data augmentation functions here """
from torchvision import transforms

""" > Import your loss functions here """
from torch import nn

""" > Import your data samplers here """
from src.samplers.instance_id_sampler import InstanceIdSampler

import src.utils.logging as logging
logger = logging.get_logger(__name__)

class BaseFactory():
    """Base factory for dataset and model generator"""
    def __init__(self):
        self.info_msg = 'Generating object'
        self.objfn_dict = None

    def generate(self, name, **kwargs):
        """Generate object based on the given name and variables
        Args:
            name: a string to describe the type of the object
            kwargs: keyworded variables of the object to generate

        Return:
            Generated object with corresponding type and arguments
        """
        assert name in self.objfn_dict, \
            '{} not recognized. ' \
            'Only support:\n{}'.format(name, self.objfn_dict.keys())
        
        logger.info('%s: %s' % (self.info_msg, name))
        logger.info('Given parameters:')
        for key, val in kwargs.items():
            logger.info('    %s = %s' % (key, val))
        logger.info('-'*80)


class ModelFactory(BaseFactory):
    """Factory for model generator"""
    def __init__(self):
        self.info_msg = 'Generating model'
        self.objfn_dict = {
            'DummyModel': DummyModel,
            'ImageClassifier': ImageClassifier,
            'TripletNet': TripletNet
        }

    def generate(self, model_name, **kwargs):
        """Generate model based on given name and variables"""
        super().generate(model_name, **kwargs)
        gen_model = self.objfn_dict[model_name](**kwargs)
        return gen_model


class DatasetFactory(BaseFactory):
    """Factory for dataset generator"""
    def __init__(self):
        self.info_msg = 'Generating dataset'
        self.objfn_dict = {
            'AIC20_VEHI_TYPE': AIC20_VEHI_TYPE,
            'AIC20_VEHI_REID': AIC20_VEHI_REID
        }

    def generate(self, dataset_name, **kwargs):
        """Generate dataset based on given name and variables"""
        super().generate(dataset_name, **kwargs)
        gen_dataset = self.objfn_dict[dataset_name](**kwargs)
        return gen_dataset

class DataAugmentationFactory(BaseFactory):
    """Factory for data augmentation object generator"""
    def __init__(self):
        self.info_msg = 'Generating data augmentation strategy'
        self.objfn_dict = {
            "resize": transforms.Resize,
            "center_crop": transforms.CenterCrop,
            "to_tensor": transforms.ToTensor,
            "normalize": transforms.Normalize
        }

    def generate(self, data_augment_name, kwargs):
        """Generate data augmentation strategies based on given name and variables"""
        if kwargs is not None:
            super().generate(data_augment_name, **kwargs)
            gen_data_augment = self.objfn_dict[data_augment_name](**kwargs)
        else:
            super().generate(data_augment_name)
            gen_data_augment = self.objfn_dict[data_augment_name]()
        return gen_data_augment


class LossFactory(BaseFactory):
    """Factory for loss function generator"""
    def __init__(self):
        self.info_msg = 'Generating loss function'
        self.objfn_dict = {
            "CrossEntropy": nn.CrossEntropyLoss,
            "MSE": nn.MSELoss
        }

    def generate(self, loss_function_name, **kwargs):
        """Generate loss function based on given name and variables"""
        super().generate(loss_function_name, **kwargs)
        gen_loss_fn = self.objfn_dict[loss_function_name](**kwargs)
        return gen_loss_fn

class DataSamplerFactory(BaseFactory):
    """Factory for loss function generator"""
    def __init__(self):
        self.info_msg = 'Generating data sampler'
        self.objfn_dict = {
            "InstanceIdSampler": InstanceIdSampler,
        }

    def generate(self, data_sampler_name, **kwargs):
        """Generate loss function based on given name and variables"""
        super().generate(data_sampler_name, **kwargs)
        gen_data_sampler = self.objfn_dict[data_sampler_name](**kwargs)
        return gen_data_sampler