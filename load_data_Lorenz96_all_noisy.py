from .utils import MyDataset
from torch.utils.data import DataLoader
import numpy as np

def get_dataloader(batch_size = 128, num_nodes=20, F=10, time_length=500, sd=0.1):
    '''
    torch.Size([batch_size, 10, num_nodes])
    '''
    data_list = []
    id_list = []
    group_list = []
    dir = "/home/private_user/work/data/Lorenz96_all_noisy/Lorenz96_node"+str(num_nodes)+"_F"+str(F)+"_T"+str(time_length)+"_sd"+str(sd)+".npy"

    X = np.load(dir)
    id_list = group_list = X

    dataset = MyDataset(X, id_list, group_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def get_causality(seed=42, num_nodes=20):
    dir = "/home/private_user/work/data/Lorenz96_all_noisy/Lorenz96_GC_node"+str(num_nodes)+"_seed"+str(seed)+".npy"
    return np.load(dir)