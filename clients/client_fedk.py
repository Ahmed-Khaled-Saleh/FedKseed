from optimizers.mezo_optimizer import *  # noqa: F403
from optimizers.mezo_bias_optimizer import *  # noqa: F403
from tqdm import tqdm
import os
import torch
from utils.validation import *  # noqa: F403

class Client(object):
    def __init__(self, idx, args, candidate_seeds, train_loader, eval_loader):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None
        self.task = train_loader.dataset[0]['task']

        self.device = torch.device(f'cuda:{args.device}')
        self.candidate_seeds = candidate_seeds

    def local_train_with_seed_pool(self, pulled_model, cur_round, memory_record_dic=None, probabilities=None, gradient_history=None):
        self.model = pulled_model
        self.model.to(self.device)
        
        if memory_record_dic is not None:
            torch.cuda.empty_cache()
        
        # initialize a seed pool
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        lr = float(self.args.lr)
        
        if self.args.batch_or_epoch == 'epoch':
            iter_steps = self.args.local_step * len(self.train_loader)
        else:
            iter_steps = self.args.local_step
            
        if self.args.bias_sampling:
            assert probabilities is not None
            framework = MeZOBiasOptimizer(self.model, args=self.args, lr=lr, candidate_seeds=self.candidate_seeds, probabilities=probabilities, gradient_history=gradient_history)  # noqa: F405
        else:
            framework = MeZOFramework(self.model, args=self.args, lr=lr, candidate_seeds=self.candidate_seeds)  # noqa: F405
        self.model.eval()
        with torch.no_grad():
            if self.args.batch_or_epoch == 'batch':
                    loss_total_train = 0.0
                    num_trained = 0
                    progress_bar = tqdm(range(iter_steps))
                    
            for cur_step in range(iter_steps):
                # init epoch progress bar
                if self.args.batch_or_epoch == 'epoch':
                    if cur_step % len(self.train_loader) == 0:
                        loss_total_train = 0.0
                        num_trained = 0
                        progress_bar = tqdm(range(len(self.train_loader)))
                try:
                    batch = next(self.train_iterator)
                except StopIteration:
                    self.train_iterator = iter(self.train_loader)
                    batch = next(self.train_iterator)
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device) 
                }
                logits, loss = framework.zo_step(batch, local_seed_pool=self.local_seed_pool)
                progress_bar.update(1)
                if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                    loss_total_train += loss
                    num_trained += len(batch['input_ids'])
                if self.args.batch_or_epoch == 'epoch':
                    progress_bar.set_description(f'client {self.idx} train at epoch {int(cur_step / len(self.train_loader)) + 1}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')
                else:
                    progress_bar.set_description(f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}')
        # save both CPU and GPU memory
        del framework
        # save model to disk
        # check if folder saved_models exists
        if not os.path.exists('/scratch/project_2009050/saved_models'):
            os.makedirs('/scratch/project_2009050/saved_models')
        
        torch.save(self.model.state_dict(), f'/scratch/project_2009050/saved_models/client_{self.idx}.pt')

        self.model = None
        
        if memory_record_dic is not None:
            memory_record_dic[self.device.index] = {}
            memory_record_dic[self.device.index]['max_memory_allocated'] = torch.cuda.max_memory_allocated(self.device)
            memory_record_dic[self.device.index]['max_memory_reserved'] = torch.cuda.max_memory_reserved(self.device)



    def train_error_and_loss(self):
        if self.model is None:
            self.model = torch.load(f'/scratch/project_2009050/saved_models/client_{self.idx}.pt')
        self.model.to(self.device)
        self.model.eval()
        
        loss_total_train = 0.0
        acc_total_train = 0.0
        num_eval = 0
        
        with torch.no_grad():
            for batch in self.train_loader:
                task = batch['task']
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                outputs = self.model(**batch)
                loss = outputs.loss
                output_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    max_new_tokens=128,
                    num_beams=1,
                )
                acc_total_train += rouge_score(output_ids[0][len(batch['input_ids'][0]):], batch['labels'][0], self.tokenizer)  # noqa: F405

                if torch.isnan(loss):
                    continue
                loss_total_train += loss
                num_eval += len(batch['input_ids'])

                if num_eval == 0:
                    num_eval = 1e-10

        print()
        print()

        # self.model = self.model.cpu()
        self.model = None
        return (acc_total_train / num_eval), (loss_total_train / num_eval).item()
    
    def eval_error_and_loss(self):
        if self.model is None:
            self.model = torch.load(f'/scratch/project_2009050/saved_models/client_{self.idx}.pt')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        loss_total_eval = 0.0
        acc_total_eval = 0.0
        num_eval = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                task = batch['task'][0]
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                }
                
                outputs = self.model(**batch)
                loss = outputs.loss
                if torch.isnan(loss):
                    continue
                loss_total_eval += loss

                output_ids = self.model.generate(
                    input_ids=batch['input_ids'],
                    max_new_tokens=128,
                    num_beams=1,
                )
                acc_total_eval += rouge_score(output_ids[0][len(batch['input_ids'][0]):], batch['labels'][0], self.tokenizer)  # noqa: F405

                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10

        print()
        print()

        self.model = self.model.cpu()
        return (acc_total_eval / num_eval), (loss_total_eval / num_eval).item()

    def clear_model(self):
        # clear model to same memory
        self.model = None

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