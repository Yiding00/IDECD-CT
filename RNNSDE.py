import torch
from torch import nn
import numpy as np

class TimeHomogeneousSDE(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, dims, hidden, device):
        """Initialization of TimeHomogeneousSDE
        dims:  [N, D]

        Parameters
        N: number of variables
        D: Diffusion size
        """
        super(TimeHomogeneousSDE, self).__init__()
        assert len(dims) == 2

        self.num_nodes = dims[0]
        self.diffusion_size = dims[1]

        self.f_weight = None
        self.g_weight = None
        self.f_nets = nn.ModuleList([BaseModule(hidden = hidden) for _ in range(self.num_nodes)])
        self.g_nets = nn.ModuleList([BaseModule(hidden = hidden) for _ in range(self.num_nodes)])
        self.f_nets.to(device)
        self.g_nets.to(device)


    def f(self, t, x):
        """Drift function
        x: [batch size, number of variables]
        t: [time point]

        return  f(x): [batch size, number of variables].
        """
        results = []
        for i in range(self.num_nodes):
            results.append(self.f_nets[i](x, weight=self.f_weight[i,:,:]))
        results = torch.stack(results, dim=1).squeeze(2)
        return results
    

    def g(self, t, x):
        """Diffusion function
        input   x: [batch size, number of variables]
                t: [time point]

        return  g(x): [batch size, number of variables, diffusion dimension].
        """
        results = []
        for i in range(self.num_nodes):
            results.append(self.g_nets[i](x, weight=self.g_weight[i,:,:,:]))
        results = torch.stack(results, dim=1)
        return results
    
    def l2_reg(self):
        return torch.sum(self.f_weight**2) + torch.sum(self.g_weight**2)

    def causal_graph(self, threshold=0):
        W = self.f_weight.cpu().detach().numpy()
        W = np.sqrt(W**2)
        W[np.abs(W) < threshold] = 0
        Wg = self.g_weight.cpu().detach().numpy()
        Wg = np.sum(np.sqrt(Wg**2), axis=3)
        Wg[np.abs(Wg) < threshold] = 0
        return np.round(W, 2), np.round(Wg, 2)
    

class BaseModule(nn.Module):
    def __init__(self, hidden):
        """
        Base 模块初始化

        """
        super(BaseModule, self).__init__()
        self.fc = nn.Linear(hidden, 1, bias=True)
        self.elu = nn.ELU()
        # # 显式初始化权重和偏置为0
        # nn.init.zeros_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)

    def forward(self, x, weight):
        """
        前向传播
        x: 输入数据 [batch size, number of variables]
        """
        if len(weight.shape) == 2:
            temp = x@weight
            result = self.fc(self.elu(temp))
        else:
            weight = weight.view(weight.shape[0], -1)
            temp = x@weight
            temp1 = self.fc(self.elu(temp))
            result = temp1.view(x.shape[0], -1)
        return result # f(x): [batch size, number of variables] OR g(x): [batch size, number of variables, diffusion dimension]


class GRUModule_small(nn.Module):
    def __init__(self, input_dim, nodes_num, hidden_dim, window_size, hidden):
        """
        GRU 模块初始化
        input_dim: 节点数
        hidden_dim: GRU 隐藏层维度
        window_size: 滑动窗口的长度
        hidden: 建模每一个结点因果的隐藏层维度
        """
        super(GRUModule_small, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.nodes_num = nodes_num
        self.hidden = hidden

        # GRU 模块
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.elu = nn.ELU()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, stacked_Data):
        stacked_Data_reshape = stacked_Data.view(stacked_Data.shape[0],stacked_Data.shape[1],-1)

        hidden, _ = self.gru(stacked_Data_reshape)
        hidden = self.elu(hidden)
        hidden = self.fc(hidden)
        f_weight_temp = hidden[:,:,:self.nodes_num*self.nodes_num*self.hidden]
        f_weight = f_weight_temp.view(f_weight_temp.shape[0], f_weight_temp.shape[1], self.nodes_num, self.nodes_num, self.hidden)
        g_weight_temp = hidden[:,:,self.nodes_num*self.nodes_num*self.hidden:]
        g_weight = g_weight_temp.view(g_weight_temp.shape[0], g_weight_temp.shape[1], self.nodes_num, self.nodes_num, self.hidden, -1)*1e-3
        return f_weight, g_weight
    

def proximal(w, threshold=0.01, target=0):
    """Proximal step"""
    # w shape [nodes_num, nodes_num, hidden]
    if target == 0:
        tmp = torch.sum(w**2, dim=2).pow(0.5) - threshold
        alpha = torch.clamp(tmp, min=0)
        v = torch.nn.functional.normalize(w, dim=2) * alpha[:, :, None]
        w.data=v.data
    else:
        w.data = torch.sign(w) * torch.clamp(torch.abs(w), max=1+threshold)
        w.data[torch.abs(w.data) > (1 - threshold)] = torch.sign(w.data[torch.abs(w.data) > (1 - threshold)])

class GRUModule(nn.Module):
    def __init__(self, input_dim, nodes_num, hidden_dim, hidden, changepoint_list):
        """
        GRU 模块初始化
        input_dim: 节点数
        hidden_dim: GRU 隐藏层维度
        hidden: 建模每一个结点因果的隐藏层维度
        """
        super(GRUModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nodes_num = nodes_num
        self.hidden = hidden
        self.changepoint_list = changepoint_list

        # GRU 模块
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.elu = nn.ELU()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, data):
        hidden, _ = self.gru(data)
        hidden = self.elu(hidden)
        hidden = self.fc(hidden)
        f_weight_temp = hidden[:,:,:self.nodes_num*self.nodes_num*self.hidden]
        f_weight_t = f_weight_temp.view(f_weight_temp.shape[0], f_weight_temp.shape[1], self.nodes_num, self.nodes_num, self.hidden)
        g_weight_temp = hidden[:,:,self.nodes_num*self.nodes_num*self.hidden:]
        g_weight_t = g_weight_temp.view(g_weight_temp.shape[0], g_weight_temp.shape[1], self.nodes_num, self.nodes_num, self.hidden, -1)*1e-3
        f_weight = torch.zeros(f_weight_t.shape[0], len(self.changepoint_list)-1, self.nodes_num, self.nodes_num, self.hidden).to(data.device)
        g_weight = torch.zeros(g_weight_t.shape[0], len(self.changepoint_list)-1, self.nodes_num, self.nodes_num, self.hidden, g_weight_t.shape[-1]).to(data.device)
        for i in range(len(self.changepoint_list)-1):
            f_weight[:,i,:,:,:] = torch.mean(f_weight_t[:,self.changepoint_list[i]:self.changepoint_list[i+1],:,:,:], dim=1)
            g_weight[:,i,:,:,:,:] =  torch.mean(g_weight_t[:,self.changepoint_list[i]:self.changepoint_list[i+1],:,:,:,:], dim=1)

        return f_weight, g_weight