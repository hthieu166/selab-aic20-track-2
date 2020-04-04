"""Validation/Testing routine"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch

from src.utils.misc import MiscUtils
import src.utils.logging as logging
from src.inferences.base_infer import BaseInfer
logger = logging.get_logger(__name__)
import ipdb

def test(model, criterion, loaders, device, infer_fn = BaseInfer()):
    """Evaluate the performance of a model

    Args:
        model: model to evaluate
        criterion: loss function
        loader: dictionary of data loaders for testing
        device: id of the device for torch to allocate objects
        infer_fn: BaseInference object: calculate additional metrics, saving predictions 
    Return:
        test_loss: average loss over the test dataset
        test_score: score over the test dataset
    """
    # Switch to eval mode
    model.eval()

    # Setup progressbar
    pbar = MiscUtils.gen_pbar(max_value=len(test_loader), msg='Evaluate: ')

    test_loss = 0.0
    infer_fn.init_metric()

    with torch.no_grad():
        for i, (samples, labels) in enumerate(test_loader):
            # Place data on the corresponding device
            samples = samples.to(device)
            labels = labels.to(device)

            # Forwarding
            outputs = model(samples)
            loss = criterion(outputs, labels)
            test_loss += loss
            
            # Statistics
            infer_fn.batch_update(outputs, labels)
            
            # Monitor progress
            pbar.update(i+1, loss=loss.item())

    pbar.finish()
    test_loss /= len(test_loader)
    logger.info('Validation loss: %.4f' % test_loss)
    test_acc = infer_fn.finalize_metric(logger, test_loss)
    # !!! test_loss is used to choose the best model!
    return test_loss, test_acc
