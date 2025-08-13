import scipy.io
import numpy as np
from .utils import MyWindowedDataset
from torch.utils.data import DataLoader

def get_dataloader(batch_size = 1):
    '''
    torch.Size([batch_size, 10, num_nodes])
    '''
    window_num = 54
    id_list = ['4799','4820','4847','5675','5680','5710']
    data_list = []
    changepoint_list = []
    for id in id_list:
        data_list_id = np.empty((0, 25))
        timelength_list_id = []
        dir = "/home/private_user/work/data/starplus/starplus-0"+id+"-avg.mat"
        mat_data = scipy.io.loadmat(dir)
        for i in range(window_num):
            timelength_list_id.append(mat_data['data_avg'][i,0].shape[0])
            data_list_id = np.concatenate((data_list_id, mat_data['data_avg'][i,0]), axis=0)
        data_id = np.array(data_list_id)
        timelength_id = np.array(timelength_list_id)
        data_list.append(data_id)
        changepoint_list.append(timelength_id)
    data_list = np.array(data_list)
    dataset = MyWindowedDataset(data_list, id_list, changepoint_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_timelength_list():
    window_num = 54
    dir = "/home/private_user/work/data/starplus/starplus-04799-avg.mat"
    mat_data = scipy.io.loadmat(dir)
    timelength_list_id = []
    for i in range(window_num):
        timelength_list_id.append(mat_data['data_avg'][i,0].shape[0])
    timelength_id = np.array(timelength_list_id)
    return timelength_id

def get_ROI_names():
    '''
    Get the names of the ROIs in the StarPlus dataset.
    '''
    dir = "/home/private_user/work/data/starplus/starplus-05710-avg.mat"
    mat_data = scipy.io.loadmat(dir)
    ROI_names = mat_data['target_ROIs'].flatten()
    return ROI_names