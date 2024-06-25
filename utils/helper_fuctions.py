import random
import importlib
import csv
import yaml
import numpy as np
import torch
import pandas as pd

def dict_to_dataframe(dictionary):
    """
    Convert a dictionary with tuple keys (client_index, task) to a Pandas DataFrame.
    
    Parameters:
    - dictionary: dict, the dictionary to be converted
    
    Returns:
    - df: pandas DataFrame, the resulting DataFrame
    """
    # Convert dictionary to a list of tuples (client_index, task, value)
    data = [(key[0], key[1], value) for key, value in dictionary.items()]
    
    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Client Index', 'Task', 'Value'])
    
    return df


def softmax(vec):
    vec = vec - np.max(vec)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def min_max_norm(vec):
    min_val = np.min(vec)
    return (vec - min_val) / (np.max(vec) + 1e-10 - min_val)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def get_client_indices_rounds(args):
    client_indices_rounds = []
    for _ in range(args.rounds):
        client_indices_rounds.append(np.random.choice(np.arange(args.num_clients), size=int(args.num_clients * args.m), replace=False))
    return client_indices_rounds

def get_client_list(args, candidate_seeds, list_train_loader, list_eval_loader):
    Client = get_class('clients.client_' + args.name, 'Client')
    client_list = []
    for idx in range(args.num_clients):
        client_list.append(Client(idx, args, candidate_seeds, list_train_loader[idx], list_eval_loader[idx]))
    return client_list


def get_server(args, eval_loader, candidate_seeds, log_dir):
    Server = get_class('servers.server_' + args.name, 'Server')
    return Server(args, eval_loader=eval_loader, candidate_seeds=candidate_seeds, log_dir=log_dir)
