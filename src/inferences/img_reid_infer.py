from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as osp

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import numpy as np
import pandas as pd
import torch
from src.inferences.base_infer import BaseInfer
from src.utils.reid_metrics import reid_evaluate
from src.utils.misc import MiscUtils
import ipdb
class ImgReIdInfer(BaseInfer):
    """    
       Note: this class is used for image classification task
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def init_metric(self, **kwargs):
        """
            This function is called once before evaluation
        """
        #getting some new parameters #TODO: change base and cls class
        super().init_metric(**kwargs)
        loaders = kwargs['loaders']
        self.gal_ld = loaders['gallery']
        self.que_ld = loaders['query']
        self.logger = kwargs['logger']
        
        self.eval_loader = self.gal_ld
        self.eval_mess = "Embedding gallery imgs: " 
        
        # Setup progressbar
        pbar = MiscUtils.gen_pbar(max_value=len(self.que_ld), msg='Embedding queries: ')
    
        self.que_lbl = []
        self.que_emb = []
        self.gal_lbl = []
        self.gal_emb = []

        with torch.no_grad():
            for i, (samples, labels) in enumerate(self.que_ld):
                samples = samples.to(self.device)
                self.que_emb.append(self.model(samples))
                self.que_lbl.append(labels)
                #Monitor progress
                pbar.update(i+1)
        pbar.finish()
        
        self.que_emb = torch.cat(self.que_emb, dim = 0)
        self.que_lbl = torch.cat(self.que_lbl, dim = 0).detach().numpy()

    def batch_evaluation(self, samples, labels):
        """
            This function is called every batch evaluation
        """ 
        outputs = super().batch_evaluation(samples, labels)
        # Collecting all gallery embeddings
        self.gal_emb.append(outputs)
        self.gal_lbl.append(labels)


    def finalize_metric(self):
        """
            This function is called at the end of evaluation process
            Final statistic results are given
        """
        self.gal_lbl = torch.cat(self.gal_lbl, dim = 0).cpu().detach().numpy()
        self.gal_emb = torch.cat(self.gal_emb, dim = 0)
        self.idcs, mAP, cmc = reid_evaluate(self.que_emb, self.gal_emb, self.que_lbl, self.gal_lbl)
        self.logger.info('$$$ Validation mAP: %.4f' % mAP)
        self.logger.info('$$$ Validation cmc: %.4f' % cmc)
        return mAP

    def export_output(self):
        if (self.output_dir == None or self.write_output == False):
            return 
        super().export_output()
        #write embs
        self.gal_emb = self.gal_emb.cpu().detach().numpy()
        self.que_emb = self.que_emb.cpu().detach().numpy()
        np.save(osp.join(self.output_dir, "que_emb.npy"), self.que_emb)
        np.save(osp.join(self.output_dir, "gal_emb.npy"), self.gal_emb)
        #generate submission file
        que_fname = self.que_ld.dataset.get_img_names()
        gal_fname = self.gal_ld.dataset.get_img_names()
        self.logger.info("Saving embeddings")
        que_fname = np.array([int(i) for i in que_fname]).astype(np.int32)
        gal_fname = np.array([int(i) for i in gal_fname]).astype(np.int32)
        self.logger.info("Saving submission file")
        out_file  = osp.join(self.output_dir, "track2.txt")
        np.savetxt(out_file, gal_fname[self.idcs], 
                delimiter = " ", fmt = "%d", newline='\n')
