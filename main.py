import os
import time
import json
import argparse
import yaml
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from copy import deepcopy
from data.utils_data.load_data import get_datasets
from trainers.trainer import Trainer
from optimizers.mezo_torch import MeZOOptimizer
from clients.model import GPT2
from trainers.callbacks import empty_cach, log_memory
from utils.helper_fuctions import (setup_seed,  
                                   load_config,
                                   get_server,
                                   get_client_list,
                                   get_client_indices_rounds
                                   )


def local_training(client_list, 
                   memory_record_dic):
    
    lst_clients_metrics = []

    for client in client_list:
        trainer = Trainer(client)
        
        local_iters = 1
        epochs = 10
        train_loss, val_loss = trainer.train(fed= False,
                                     epochs=epochs,
                                     local_iters=local_iters,
                                     memory_record_dic=memory_record_dic,
                                     callbacks=[empty_cach, log_memory])
                
        client.train_stats['train_loss'], client.train_stats['val_loss'], client.train_stats['task'] = train_loss, val_loss, client.task
        lst_clients_metrics.append(client.train_stats)

        client.model = None

    return lst_clients_metrics

def federated_training(client_list,  
                       client_indices_rounds,
                       server,
                       args, 
                       run, 
                       memory_record_dic):
    
    lst_global_metrics_dfs = []
    for t in range(1, args.rounds + 1):
        selected_client = [client_list[i] for i in client_indices_rounds[t-1]]
        
        lst_global_metrics = []
        
        for client in selected_client:
            trainer = Trainer(client)
        
            local_iters = client.args.local_step
            epochs = 1
            
            client.model = deepcopy(server.model)
            
            metrics = {}
            train_loss, val_loss, train_acc, val_acc = trainer.train(fed= True,
                                                 epochs= epochs,
                                                 local_iters= local_iters,
                                                 memory_record_dic= memory_record_dic,
                                                 callbacks=[empty_cach, log_memory])
            
            train_loss = np.array(train_loss).mean()
            task = client.task if isinstance(client.task, str) else client.task[0]

            metrics['train_loss'], metrics['val_loss'], metrics['task'], metrics['train_acc'], metrics['val_acc'] =\
                  train_loss, val_loss, task, train_acc, val_acc
            
            lst_global_metrics.append(metrics)
        
        round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))
        run.log({f"round {t} (GM) Metrics": round_global_metrics})
        
        lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

        server.aggregate_seed_pool(selected_client)
        server.update_global_model_by_seed_pool()

    return lst_global_metrics_dfs

def process_main(args_config_fname):
    fname = args_config_fname.fname
    config = load_config(fname)
    experiment = config[0]  # Assuming single experiment per config file

    args = argparse.Namespace(**experiment)
    run = wandb.init(project=args.project, name= args.name, config=args)

    time_stamp = str(time.time())
    memory_record_dic = {}
    
    previous_metric = args.eval_metric
    args.eval_metric = 'loss'
    # set CUDA visibility to targeted cuda device, to avoid the several hundred MB memory consumption of device 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    setup_seed(args.seed)

    loss_ds, gener_ds = get_datasets(args)
    list_train_ds, list_eval_ds, tokenizer, datacollator = loss_ds
    list_train_ds_genr, list_eval_ds_genr, _, _ = gener_ds
    
    if args.dataset == 'instruct':
        args.iid = 'meta'
    log_dir = time_stamp

    if args.log_root != '':
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir)
    config = yaml.dump(args, None)
    config = '\n'.join(config.split('\n')[1:])
    print('Configs: ')
    print(config)
    print('=====================')

    # since only CUDA device is available, load all models on device 0
    args.device = 0
    client_indices_rounds = get_client_indices_rounds(args)
    client_list = []
    
    # sample `K` candidate seeds
    candidate_seeds = np.random.randint(1, 100000000000, args.K)
    server = get_server(args, candidate_seeds, log_dir)

    optimizer = MeZOOptimizer(server.model.parameters(),
                              lr= float(args.lr),
                              zo_eps= args.zo_eps,
                              candidate_seeds= candidate_seeds,
                              weight_decay= args.weight_decay)
    
    def criterion(out):
        return out.loss

    client_list = get_client_list(list_train_ds, list_eval_ds, server.model, criterion, optimizer, list_train_ds_genr, list_eval_ds_genr, tokenizer, datacollator, args, candidate_seeds)
    

    if args.log:
        with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
            json.dump(memory_record_dic, writer)


    lst_local_metrics = local_training(client_list= client_list, server=server, memory_record_dic= memory_record_dic)
    table_local_metrics = wandb.Table(dataframe=pd.DataFrame(lst_local_metrics))
    run.log({"Local Metrics": table_local_metrics})

    lst_global_metrics_dfs = federated_training(client_list, client_indices_rounds, server, args, run, memory_record_dic)

    
    
    
    

    # # reset seed to have an eval_loader with the same data samples
    # args.eval_metric = previous_metric
    # setup_seed(args.seed)
    # _, eval_loader_final, _ = get_loaders(args, only_eval=True)
    # server.eval_loader = eval_loader_final
    # eval_result = server.eval(cur_round=args.rounds, eval_avg_acc=eval_avg_acc)
    
    # if args.log:
    #     with open(os.path.join(log_dir, 'final_eval.json'), 'w') as writer:
    #         json.dump({
    #             f'final_eval_{args.eval_metric}': eval_result
    #         }, writer)
    
    # print(f'final round {args.eval_metric}: {eval_result}')
    # run.log({"Final Global Rouge":eval_result})
    run.finish()



if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    load_dotenv()
    key = os.getenv("WANDB_API_KEY")
    wandb.login(key=key, verify=False)

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--fname', type=str,
        help='name of config file to load',
        default='configs.yaml')
    
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=1, type=int,
                        help='number of nodes')
    
    args_config_fname = parser.parse_args()
    process_main(args_config_fname)