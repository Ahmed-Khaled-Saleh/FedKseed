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

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def process_main(args_config_fname):
    fname = args_config_fname.fname
    config = load_config(fname)
    experiment = config[0]  # Assuming single experiment per config file

    args = argparse.Namespace(**experiment)
    run = wandb.init(project=args.project, name= args.name, config=args)

    time_stamp = str(time.time())
    eval_avg_acc = []
    memory_record_dic = {}
    
    previous_metric = args.eval_metric
    args.eval_metric = 'loss'
    # set CUDA visibility to targeted cuda device, to avoid the several hundred MB memory consumption of device 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    setup_seed(args.seed)

    list_train_loader, eval_loader, _ = get_loaders(args)
    
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
    client_list = get_client_list(args, candidate_seeds, list_train_loader, eval_loader)

    server = get_server(args, eval_loader, candidate_seeds, log_dir)
    
    eval_result, _ = server.eval(cur_round=0, eval_avg_acc=eval_avg_acc)
    eval_avg_acc.append(eval_result)

    if args.log:
        with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
            json.dump(memory_record_dic, writer)
        with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
            json.dump({
                'eval_avg_acc': eval_avg_acc
            }, writer)


    lst_clients_metrics = []
    lst_global_metrics = []

    for client in client_list:
        metrics = {}
        local_loss = client.full_local_train(server.model_w0)
        task = client.task if isinstance(client.task, str) else client.task[0]

        metrics["task"] = task
        metrics["train_loss"] = local_loss
        lst_clients_metrics.append(metrics)
        client.model = None
    
    clients_local_metrics = wandb.Table(dataframe=pd.DataFrame(lst_clients_metrics))
    run.log({"Local Metrics":clients_local_metrics})
        


    lst_global_metrics_dfs = []
    for r in range(1, args.rounds + 1):
        run.watch(server.model)
        selected_client = [client_list[i] for i in client_indices_rounds[r-1]]
        
        if args.bias_sampling:
            probabilities = server.calculate_probabilities()
        else:
            probabilities = None
        
        for client in selected_client:
            metrics = {}
            
            train_loss = client.local_train_with_seed_pool(deepcopy(server.model), cur_round=r, memory_record_dic=memory_record_dic, probabilities=probabilities, gradient_history=server.gradient_history)
            
            task = client.task if isinstance(client.task, str) else client.task[0]
            metrics['train_loss'], metrics['task']  = train_loss, task
            lst_global_metrics.append(metrics)
            
            client.model = None

        round_global_metrics = wandb.Table(dataframe=pd.DataFrame(lst_global_metrics))
        run.log({f"round {r} Global Metrics":round_global_metrics})
        
        lst_global_metrics_dfs.append(pd.DataFrame(lst_global_metrics))

        
        # step 6, 7 
        server.aggregate_seed_pool(selected_client)

        # eqation 4  
        # server gets the latest global model from the accumulated scalar gradients
        server.update_global_model_by_seed_pool()

        # eval_result, loss_per_task = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
        # run.log({"global_loss":eval_result})
        # lst_global_metrics.append(loss_per_task)
        # eval_avg_acc.append(eval_result)

        if args.log:
            with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
                json.dump(memory_record_dic, writer)
            with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
                json.dump({
                    'eval_avg_acc': eval_avg_acc
                }, writer)
    
    df = pd.concat(lst_global_metrics_dfs, ignore_index=True)
    df.to_csv(os.path.join(log_dir, 'global_metrics.csv'), index=False)
    global_metrics_table = wandb.Table(dataframe=df)
    run.log({"All Global Metrics":global_metrics_table})

    # df_global = pd.DataFrame(lst_global_metrics)
    # df_global.to_csv(os.path.join(log_dir, 'global_metrics.csv'), index=False)
    # global_metrics_table = wandb.Table(dataframe=df_global)
    # run.log({"Global Metrics":global_metrics_table})

    # reset seed to have an eval_loader with the same data samples
    args.eval_metric = previous_metric
    setup_seed(args.seed)
    _, eval_loader_final, _ = get_loaders(args, only_eval=True)
    server.eval_loader = eval_loader_final
    eval_result = server.eval(cur_round=args.rounds, eval_avg_acc=eval_avg_acc)
    
    if args.log:
        with open(os.path.join(log_dir, 'final_eval.json'), 'w') as writer:
            json.dump({
                f'final_eval_{args.eval_metric}': eval_result
            }, writer)
    
    print(f'final round {args.eval_metric}: {eval_result}')
    run.log({"Final Global Rouge":eval_result})
    run.finish()



if __name__ == '__main__':

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