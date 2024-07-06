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
        # if memory_record_dic is not None:
        #     torch.cuda.empty_cache()

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
        # losses = mp.spawn(self.local_train_with_seed_pool, 
        #                   args=(pulled_model, cur_round, memory_record_dic, total_epochs), 
        #                   nprocs=self.args.world_size)
        self.local_train_with_seed_pool(pulled_model, cur_round, memory_record_dic, total_epochs)
        return self.train_losses[0]
        # self.model.eval()
        # loss_total_train = 0.0
        # num_trained = 0
        # progress_bar = tqdm(range(iter_steps))

        # for cur_step in range(iter_steps):
        #     # init epoch progress bar
        #     if self.args.batch_or_epoch == 'epoch':
        #         if cur_step % len(self.train_loader) == 0:
        #             loss_total_train = 0.0
        #             num_trained = 0
        #             progress_bar = tqdm(range(len(self.train_loader)))

        #     try:
        #         batch = next(self.train_iterator)
        #     except StopIteration:
        #         self.train_iterator = iter(self.train_loader)
        #         batch = next(self.train_iterator)

        #     batch = {
        #         'input_ids': batch['input_ids'],
        #         'labels': batch['labels'],
        #         'attention_mask': batch['attention_mask']
        #     }

        #     def closure():
        #         return self.model(**batch)

        #     loss = optimizer.step(closure)

        #     progress_bar.update(1)
        #     if not torch.isnan(loss):
        #         loss_total_train += loss.item()
        #         num_trained += len(batch['input_ids'])

        #     if self.args.batch_or_epoch == 'epoch':
        #         progress_bar.set_description(f'client {self.idx} train at epoch {int(cur_step / len(self.train_loader)) + 1}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')
        #     else:
        #         progress_bar.set_description(f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')

        # if num_trained == 0:
        #     num_trained = 1e-10
        # train_loss = loss_total_train / num_trained

        # if memory_record_dic is not None:
        #     memory_record_dic[self.device.index] = {}
        #     memory_record_dic[self.device.index]['max_memory_allocated'] = torch.cuda.max_memory_allocated(self.device)
        #     memory_record_dic[self.device.index]['max_memory_reserved'] = torch.cuda.max_memory_reserved(self.device)

        # return train_loss

    # def eval_error_and_loss(self, tokenizer):

    #     self.model
    #     self.model.eval()
        
    #     loss_total_train = 0.0
    #     acc_total_train = 0.0
    #     num_eval = 0
        
    #     with torch.no_grad():
    #         for batch in self.eval_loader:
    #             batch = {
    #                 'input_ids': batch['input_ids'],
    #                 'labels': batch['labels'],
    #                 'attention_mask': batch['attention_mask']
    #             }
    #             outputs = self.model(**batch)
    #             loss = outputs.loss

    #             if torch.isnan(loss):
    #                 continue
    #             loss_total_train += loss
    #             num_eval += len(batch['input_ids'])

    #             if num_eval == 0:
    #                 num_eval = 1e-10

    #     print()
    #     print()

    #     # self.model = self.model.cpu()
    #     # self.model = None
    #     return (acc_total_train / num_eval), (loss_total_train / num_eval).item()
    
   
    # def clear_model(self):
    #     # clear model to same memory
    #     self.model = None

    # def migrate(self, device):
    #     """
    #     migrate a client to a new device
    #     """
    #     self.device = device

    # def pull(self, forked_global_model):
        # """
        # pull model from the server
        # """
        # self.model = forked_global_model