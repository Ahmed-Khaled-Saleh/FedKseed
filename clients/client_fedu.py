from optimizers.mezo_optimizer import *  # noqa: F403
from optimizers.mezo_bias_optimizer import *  # noqa: F403
from tqdm import tqdm
import os
import numpy as np
import torch
import copy
from utils.validation import *  # noqa: F403
from optimizers.mezo_torch import MeZOOptimizer
from trainers.trainer import Trainer
from clients.base_client import BaseClient
from copy import deepcopy
class Client_fedu(BaseClient):
    def __init__(self,
                 train_ds,
                 eval_ds,
                 model,
                 criterion,
                 optimizer,
                 train_ds_genr,
                 eval_ds_genr,
                 tokenizer,
                 datacollator,
                 idx,
                 args,
                 candidate_seeds, 
                 beta,
                 L_k,
                 K):

        '''
        A client is defined as an object that contains :

        1- **Essentials**:
            dataseet (train, eval), model, loss function (criterion), and an optimizer.
            
        2- **Extra information**:
            Task-dependent and algrithm-specifics
        '''
        
        super().__init__(train_ds, eval_ds, model, criterion, optimizer)

        self.train_ds_genr = train_ds_genr
        self.eval_ds_genr = eval_ds_genr
        self.tokenizer = tokenizer
        self.data_collator = datacollator
        self.idx = idx
        self.args = args
        self.device = torch.device(f'cuda:{self.args.device}')
        self.candidate_seeds = candidate_seeds
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        self.task = self.dataset[0]['task']
        self.task = self.task if isinstance(self.task, str) else self.task[0]
        self.train_stat = {}
        self.test_stats = {}
        self.L_k = L_k
        self.beta = beta
        self.K = K
        self.learning_rate = float(self.args.lr)

    def aggregate_parameters(self, user_list, alk_connection):
        avg_weight_different = copy.deepcopy(list(self.model.parameters()))
        akl = alk_connection
        for param in avg_weight_different:
            param.data = torch.zeros_like(param.data)
        
        # Calculate the diffence of model between all users or tasks
        for i in range(len(user_list)):
            if(self.idx != user_list[i].idx):
                if(self.K > 0 and self.K <= 2):
                    akl[int(self.idx)][int(user_list[i].idx)] = self.get_alk()
                
                l_model = deepcopy(self.model)
                model_path = os.path.join(user_list[i].model_path, f"client_ + {user_list[i].idx}.pt")
                l_model.load_state_dict(torch.load(model_path))
                
                for avg, current_task, other_tasks in zip(avg_weight_different, self.model.parameters(), l_model.parameters()):
                    avg.data += akl[int(self.idx)][int(user_list[i].idx)] * (current_task.data.clone() - other_tasks.data.clone())
            
            l_model = None
        for avg, current_task in zip(avg_weight_different, self.model.parameters()):
            current_task.data = current_task.data - 0.5 * self.learning_rate * self.L_k * self.beta * self.local_epochs * avg

    def clear_model(self):
        self.model = None

    def _add_seed_pole(self, zo_random_seed, projected_grad):
        if self.local_seed_pool is not None:
            self.local_seed_pool[zo_random_seed] += projected_grad
    
    def migrate(self, device):
        """
        migrate a client to a new device
        """
        self.device = device

    def pull(self, forked_global_model):
        """
        pull model from the server
        """
        self.model = forked_global_model

    def get_alk(self):
        # temporary fix value of akl, all client has same value of akl
        #akl = 0.25 # can set any value but need to modify eta accordingly
        akl = 0.5
        #akl = 1
        return akl
    
    def save_model(self):
        self.model_path = os.path.join("models", self.idx)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        torch.save(self.model, os.path.join(self.model_path, "client_" + self.idx + ".pt"))
