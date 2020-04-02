"""Main code for train/val/test"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from src.utils.load_cfg import ConfigLoader
from src.factories import ModelFactory, DatasetFactory, DataAugmentationFactory, LossFactory, DataSamplerFactory 
from trainer import train
from tester import test
import src.utils.logging as logging

from torchvision import transforms
logger = logging.get_logger(__name__)

import ipdb
from src.inferences.img_cls_infer import ImgClsInfer

def parse_args():
    """Parse input arguments"""
    def str2bool(v):
        """Convert a string to boolean type"""
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--model_cfg', type=str,
        help='Path to the model config filename')
    parser.add_argument(
        '-d', '--dataset_cfg', type=str,
        help='Path to the dataset config filename')
    parser.add_argument(
        '-t', '--train_cfg', type=str,
        help='Path to the training config filename')

    parser.add_argument(
        '-i', '--is_training', type=str2bool,
        help='Whether is in training or testing mode')
    parser.add_argument(
        '-m', '--train_mode', type=str,
        choices=['from_scratch', 'from_pretrained', 'resume'],
        help='Which mode to start the training from')
    parser.add_argument(
        '-l', '--logdir', type=str,
        help='Directory to store the log')
    parser.add_argument(
        '--log_fname', type=str, default=None,
        help='Path to the file to store running log (beside showing to stdout)')

    parser.add_argument(
        '-w', '--num_workers', type=int, default=4,
        help='Number of threads for data loading')
    parser.add_argument(
        '-g', '--gpu_id', type=int, default=-1,
        help='Which GPU to run the code')

    parser.add_argument(
        '--pretrained_model_path', type=str, default='',
        help='Path to the model to test. Only needed if is not training or '
             'is training and mode==from_pretrained')

    parser.add_argument(
        '--is_data_augmented', type=str2bool,
        help='Whether training set is augmented')
    
    parser.add_argument(
        '--pred_path', type=str, default=None,
        help='Path to save predictions when infer testing data')
    
    #### Only for finetuning by subject ID ####
    parser.add_argument(
        '--datalst_pth', type=str, default=None,
        help='Specify the path to msrs grouped by subject ID for finetuning')

    args = parser.parse_args()

    if (not args.is_training) or \
            (args.is_training and args.train_mode == 'from_pretrained'):
        assert os.path.isfile(args.pretrained_model_path), \
            'pretrained_model_path not found: {}'.format(args.pretrained_model_path)
    return args


def main():
    """Main function"""
    # Load configurations
    model_name, model_params = ConfigLoader.load_model_cfg(args.model_cfg)
    dataset_name, dataset_params = ConfigLoader.load_dataset_cfg(args.dataset_cfg)
    train_params = ConfigLoader.load_train_cfg(args.train_cfg)
    
    # Copy over some parameters


    # Set up device (cpu or gpu)
    if args.gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.gpu_id)
    logger.info('Using device: %s' % device)
    
    # Prepare datalst_pth
    if (args.datalst_pth is not None):
        for lst in dataset_params['datalst_pth'].keys():
            dataset_params['datalst_pth'][lst] = \
            os.path.join(args.datalst_pth, dataset_params['datalst_pth'][lst])

    # Set up common parameters for data loaders (shared among train/val/test)
    dataset_factory = DatasetFactory()
    loader_params = {
        'batch_size': train_params['batch_size'],
        'num_workers': args.num_workers,
    }

    # Build model
    model_factory = ModelFactory()
    model = model_factory.generate(model_name, device=device, **model_params)
    model = model.to(device)
    
    # Data augmentation
    if args.is_data_augmented:
        data_augmentation_factory = DataAugmentationFactory()
        composed_transforms = transforms.Compose([
            data_augmentation_factory.generate(
                i, train_params['data_augment'][i])
            for i in train_params['data_augment'].keys()])
        dataset_params['transform'] = composed_transforms
    else:
        dataset_params['transform'] = None
    
    # Set up loss criterion
    loss_fn_factory = LossFactory()
    criterion = loss_fn_factory.generate(train_params['loss_fn'])

    # Set up infer funtions
    infer_fn = ImgClsInfer()
    # Main pipeline
    if args.is_training:
        # Create data loader for training
        train_dataset = dataset_factory.generate(
            dataset_name, mode='train', **dataset_params) 
        # Create data sampler obj if neccessary
        if ('batch_sampler' in train_params):
            smplrFact = DataSamplerFactory()
            sampler_params = list(train_params['batch_sampler'].values())[0]
            sampler_params['dataset'] = train_dataset
            train_sampler = smplrFact.generate(
                list(train_params['batch_sampler'].keys())[0],
                **sampler_params
            )
            train_loader = DataLoader(train_dataset, 
            batch_sampler=train_sampler, num_workers= args.num_workers)
        else:
            train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True, **loader_params)

        # Create data loader for validation
        val_dataset = dataset_factory.generate(
            dataset_name, mode='val', **dataset_params)
        val_loader = DataLoader(
            val_dataset, shuffle=False, drop_last=False, **loader_params)

        # Create optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_params['init_lr'],
            momentum=train_params['momentum'],
            weight_decay=train_params['weight_decay'],
            nesterov=True)

        # Train/val routine
        train(model, optimizer, criterion, train_loader, val_loader, args.logdir,
              args.train_mode, train_params, device, args.pretrained_model_path, infer_fn)
    else:
        # Create data loader for testing
        test_dataset = dataset_factory.generate(
            dataset_name, mode='test', **dataset_params)
        test_loader = DataLoader(
            test_dataset, shuffle=False, drop_last=False, **loader_params)

        # Test routine
        model.load_model(args.pretrained_model_path)
        test(model, criterion, test_loader, device, infer_fn)
        
    return 0


if __name__ == '__main__':
    # Fix random seeds here for pytorch and numpy
    torch.manual_seed(1)
    np.random.seed(2)

    # Parse input arguments
    args = parse_args()

    # Setup logging format
    logging.setup_logging(args.log_fname)

    # Run the main function
    sys.exit(main())
