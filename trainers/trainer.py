import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        client
    ) -> None:
        
        '''
            The Trainer class implements the training loop with the train() function.
            It accepts a client at initialization, which contains all necessary infromation 
            to implement a training loop for a federated learning setup.
        '''
        
        self.client = client
        
        
    def _run_batch(self, batch):
        self.client.optimizer.zero_grad()
        def closure():
            out = self.client.model(**batch)
            return self.client.criterion(out)

        loss, zo_random_seed, projected_grad = self.client.optimizer.step(closure)
        self.client._add_seed_pole(zo_random_seed, projected_grad)

        if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
            return loss
        return 0
    
    def _run_epoch(self):
        total_loss = 0
        progress_bar = tqdm(range(len(self.client.train_loader)))

        with torch.inference_mode():
            for i, batch in enumerate(self.client.train_loader):

                batch = {
                    'input_ids': batch['input_ids'].to(self.client.device),
                    'labels': batch['labels'].to(self.client.device),
                    'attention_mask': batch['attention_mask'].to(self.client.device) 
                }

                loss = self._run_batch(batch)
                if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
                    continue
                total_loss += loss
                
                if i % 1000 == 999:
                    last_loss = total_loss / 1000 
                    progress_bar.update(i)
                    progress_bar.set_description(f'client {self.client.idx} Fuly Local Training , loss: {last_loss}')
    
        return total_loss / len(self.client.train_loader)
    

    def _run_epoch_fed(self, local_iters):
        total_loss = 0
        progress_bar = tqdm(range(local_iters))

        with torch.inference_mode():
            for r in range(local_iters):
                num_trained = 0
                try:
                    batch = next(self.client.train_iterator)
                except StopIteration:
                    self.client.train_iterator = iter(self.client.train_loader)
                    batch = next(self.client.train_iterator)
                
                batch = {
                    'input_ids': batch['input_ids'].to(self.client.device),
                    'labels': batch['labels'].to(self.client.device),
                    'attention_mask': batch['attention_mask'].to(self.client.device) 
                }
                
                loss = self._run_batch(batch)

                progress_bar.update(1)
                progress_bar.set_description(f'client {self.client.idx} train at step {r}, loss: {total_loss / num_trained if num_trained != 0 else 0.0}')

                if (not torch.isnan(loss)) and (self.client.args.grad_clip <= 0 or loss != 0.0):
                    total_loss += loss
                    num_trained += len(batch['input_ids'])

            if num_trained == 0:
                num_trained = 1e-10

        avg_round_loss = total_loss / num_trained
                
        return avg_round_loss

    
    def train(self,
              fed= False,
              epochs= 10,
              local_iters= 1,
              memory_record_dic= None,
              callbacks= []):
        
        self.client.model.to(self.client.device)

        if callbacks:
            callbacks[0](memory_record_dic)
        
        self.client.model.eval()

        val_loss = self.eval()
        
        train_losses = []
        for _ in range(epochs):

            if fed:
                avg_train_loss = self._run_epoch_fed(local_iters)
            else:
                avg_train_loss = self._run_epoch()

            train_losses.append(avg_train_loss.item())
        
        if callbacks:
            callbacks[1](memory_record_dic, self.client.device)

        return train_losses, val_loss
    
    def eval(self):
        total_loss = 0

        def _run_batch(self, batch):
            out = self.client.model(**batch)
            loss = self.client.criterion(out)
            return loss
        
        with torch.no_grad():
            for i, batch in enumerate(self.client.eval_loader):
                
                batch = {
                    'input_ids': batch['input_ids'].to(self.client.device),
                    'labels': batch['labels'].to(self.client.device),
                    'attention_mask': batch['attention_mask'].to(self.client.device) 
                }
                 
                loss = _run_batch(batch)

                if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                    continue
                total_loss += loss              

        return total_loss / len(self.client.eval_loader)
    
   
