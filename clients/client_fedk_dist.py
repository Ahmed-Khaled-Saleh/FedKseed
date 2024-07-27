from optimizers.mezo_torch import MeZOOptimizer  # noqa: F403
from tqdm import tqdm
import os
import torch
from copy import deepcopy
from utils.validation import *  # noqa: F403
from clients.distributed_trainer import (Trainer, 
                                         prepare_dataloader)
from torch.utils.data import DataLoader
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp

class Client(object):
    def __init__(self, idx, args, candidate_seeds, ds, eval_loader):
        self.idx = idx
        self.args = args
        self.train_ds, self.collat_fn = ds[0], ds[1]
        self.eval_loader = eval_loader
        self.model = None
        self.task = self.train_ds[0]['task']

        self.candidate_seeds = candidate_seeds
        self.train_losses = []
            

    def local_train_with_seed_pool(self, pulled_model, cur_round= None, memory_record_dic=None, total_epochs=1):
        self.model = pulled_model

        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        lr = float(self.args.lr)
        # if self.args.batch_or_epoch == 'epoch':
        #     iter_steps = self.args.local_step * len(self.train_loader)
        # else:
        # iter_steps = self.args.local_step

        self.optimizer = MeZOOptimizer(self.model.parameters(), 
                                       lr= lr, 
                                       zo_eps= self.args.zo_eps, 
                                       local_seed_pool= self.local_seed_pool,
                                       candidate_seeds= self.candidate_seeds,
                                       weight_decay= self.args.weight_decay)
            
        print("Entering local_train_with_seed_pool")
        self.train_loader = prepare_dataloader(self.train_ds, self.args.batch_size, self.collat_fn)
        print("About to create Trainer")
        trainer = Trainer(self.model, self.train_loader, self.optimizer)
        print("Trainer created successfully")
        epoch_losses = trainer.train(total_epochs)
        print("Training completed")
        self.train_losses.extend(epoch_losses)


    def client_DDPTrain(self, pulled_model, cur_round= None, memory_record_dic=None, total_epochs=1):
        self.args.world_size = torch.cuda.device_count()
        self.local_train_with_seed_pool(pulled_model, cur_round, memory_record_dic, total_epochs)
        return self.train_losses[0]
        