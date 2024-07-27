import os
import time
from copy import deepcopy
import json
import argparse
from dotenv import load_dotenv
import yaml
import numpy as np
import pandas as pd
import wandb

# from data.utils_data.mtl_dataset import get_loaders
from data.utils_data.load_data import get_loaders
from utils.helper_fuctions import (setup_seed,  
                                   load_config, 
                                   get_client_indices_rounds, 
                                   get_client_list,
                                   get_server,
                                   dict_to_df)
from torch.distributed import init_process_group, destroy_process_group
import torch




def ddp_setup():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        init_process_group(backend='nccl')
        print(f"Distributed mode initialized with rank {rank} and world size {world_size}")
    else:
        raise RuntimeError("RANK and WORLD_SIZE environment variables are not set")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



def fully_local_training(client_list, server):
    lst_clients_metrics = []
    for client in client_list:
        metrics = {}
        local_loss = client.client_DDPTrain(deepcopy(server.model), total_epochs= 10)
        task = client.task if isinstance(client.task, str) else client.task[0]

        metrics["task"] = task
        metrics["train_loss"] = local_loss
        lst_clients_metrics.append(metrics)
        client.model = None

    return lst_clients_metrics



def federated_training(client_list, server, 
                       client_indices_rounds,
                       args, 
                       run, 
                       memory_record_dic):
    
    lst_global_metrics = []
    lst_global_metrics_dfs = []
    for r in range(1, args.rounds + 1):
        selected_client = [client_list[i] for i in client_indices_rounds[r-1]]
        
        for client in selected_client:
            metrics = {}
            train_loss = client.client_DDPTrain(deepcopy(server.model), cur_round=r, memory_record_dic=memory_record_dic)
            train_loss = train_loss if isinstance(train_loss, float) else train_loss[0]

            task = client.task if isinstance(client.task, str) else client.task[0]
            metrics['train_loss'], metrics['task']  = train_loss, task
            lst_global_metrics.append(metrics)
            
            client.model = None

        # round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))
        # run.log({f"round {r} (GM) Metrics":round_global_metrics})
        
        lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

        server.aggregate_seed_pool(selected_client)
        server.update_global_model_by_seed_pool()

        # if args.log:
        #     with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
        #         json.dump(memory_record_dic, writer)
        #     with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
        #         json.dump({
        #             'eval_avg_acc': eval_avg_acc
        #         }, writer)

        return lst_global_metrics_dfs



def process_main(args_config_fname):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ddp_setup()
    fname = args_config_fname.fname
    config = load_config(fname)
    experiment = config[0]  # Assuming single experiment per config file

    args = argparse.Namespace(**experiment)
    # run = wandb.init(project=args.project, name= args.name, config=args)

    time_stamp = str(time.time())
    eval_avg_acc = []
    
    
    args.eval_metric = 'loss'
    setup_seed(args.seed)

    lst_train_ds, lst_eval_ds = get_loaders(args)
    
    if args.dataset == 'instruct':
        args.iid = 'meta'
    log_dir = time_stamp

    memory_record_dic = {}
    if args.log_root != '':
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir)
    
    config = yaml.dump(args, None)
    config = '\n'.join(config.split('\n')[1:])
    print('Configs: ')
    print(config)
    print('=====================')

    client_indices_rounds = get_client_indices_rounds(args)
    client_list = []
    
    candidate_seeds = np.random.randint(1, 100000000000, args.K)
    client_list = get_client_list(args, candidate_seeds, lst_train_ds, lst_eval_ds)
    server = get_server(args, lst_eval_ds, candidate_seeds, log_dir)
    # eval_result, _ = server.eval(cur_round=0, eval_avg_acc=eval_avg_acc)
    # eval_avg_acc.append(eval_result)

    if args.log:
        with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
            json.dump(memory_record_dic, writer)
        with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
            json.dump({
                'eval_avg_acc': eval_avg_acc
            }, writer)


    lst_clients_metrics = fully_local_training(client_list, server)
    # clients_local_metrics = wandb.Table(dataframe=pd.DataFrame(lst_clients_metrics))
    # run.log({"Fully Local Metrics":clients_local_metrics})
    
    # for r in range(1, args.rounds + 1):
    #     run.watch(server.model)
    #     selected_client = [client_list[i] for i in client_indices_rounds[r-1]]
        
    #     if args.bias_sampling:
    #         probabilities = server.calculate_probabilities()
    #     else:
    #         probabilities = None
        
    #     for client in selected_client:
    #         metrics = {}
    #         train_loss = client.client_DDPTrain(deepcopy(server.model), cur_round=r, memory_record_dic=memory_record_dic)
            
            
    #         task = client.task if isinstance(client.task, str) else client.task[0]
    #         metrics['train_loss'], metrics['task']  = train_loss, task
    #         lst_global_metrics.append(metrics)
            
    #         client.model = None

    #     round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))
    #     run.log({f"round {r} (GM) Metrics":round_global_metrics})
        
    #     lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

    #     server.aggregate_seed_pool(selected_client)
    #     server.update_global_model_by_seed_pool()

    #     if args.log:
    #         with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
    #             json.dump(memory_record_dic, writer)
    #         with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
    #             json.dump({
    #                 'eval_avg_acc': eval_avg_acc
    #             }, writer)
    
    # lst_global_metrics_dfs = federated_training(client_list, server, client_indices_rounds, args, run, memory_record_dic)
    # df = pd.concat(lst_global_metrics_dfs, ignore_index=True)
    # # df.to_csv(os.path.join(log_dir, 'global_metrics.csv'), index=False)
    # global_metrics_table = wandb.Table(dataframe=df)
    # run.log({"Train loss across client (GM)":global_metrics_table})

    # if args.log:
    #     with open(os.path.join(log_dir, 'final_eval.json'), 'w') as writer:
    #         json.dump({
    #             f'final_eval_{args.eval_metric}': eval_result
    #         }, writer)
    
    # print(f'final round {args.eval_metric}: {eval_result}')
    # run.log({"Final Global Rouge":eval_result})
    # run.finish()
    destroy_process_group()

if __name__ == "__main__":
    load_dotenv()
    # key = os.getenv("WANDB_API_KEY")
    # wandb.login(key=key, verify=False)

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