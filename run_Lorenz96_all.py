def main(config=None):
    # from Modules.LoadData.load_data_Lorenz96_all import get_dataloader, get_causality
    from Modules.LoadData.load_data_Lorenz96_all_noisy import get_dataloader, get_causality
    import numpy as np
    import torch
    from . import RNNSDE as models
    from torch.nn import functional as F
    from torchsde import sdeint
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    from sklearn.metrics import roc_auc_score
    import torch.optim.lr_scheduler as lr_scheduler
    from torch import nn
    
    default_config = {
        'num_nodes': 20,
        'time_length': 500,
        'window_size': 100,
        'hidden': 10,
        'causality_std': 1e-4,
        'horizon': 10,
        'stride': 1,
        'diffusion_size': 1,
        'epoch': 10000,
        'lr': 2e-5,
        'reg': 1e-2,
        'target': 0,
        'threshold': 0,
        'show_graph': False,
        'device': 'cuda:1',
        'scheduler': 'CosineAnnealingLR',
        'sd': 0.1
    }
    if config:
        default_config.update(config)
    window_size=default_config['window_size']
    hidden=default_config['hidden']
    causality_std=default_config['causality_std']
    horizon=default_config['horizon']
    stride=default_config['stride']
    diffusion_size=default_config['diffusion_size']
    epoch=default_config['epoch']
    lr=default_config['lr']
    reg=default_config['reg']
    target=default_config['target']
    threshold=default_config['threshold']
    show_graph=default_config['show_graph']
    device=default_config['device']
    num_nodes=default_config['num_nodes']
    time_length=default_config['time_length']
    scheduler_name=default_config['scheduler']
    data_loader = get_dataloader(batch_size = 1, num_nodes=num_nodes, time_length=time_length, sd=default_config['sd'])
    for data, id, group in data_loader:

        real_causality=get_causality(num_nodes=num_nodes)

        data = data.to(device)  # [batch_size, time_length, nodes_num]
        batch_size, time_length, nodes_num = data.shape

        window_num = time_length // window_size
        stacked_data = torch.stack([
            data[:, j*window_size:(j+1)*window_size, :]
            for j in range(window_num)
            ], dim=1)  # [batch_size, window_num, window_size, nodes_num]
        horizoned_data = torch.stack([
            stacked_data[:, :, k*stride:k*stride+horizon, :]
            for k in range((window_size-horizon)//stride+1)
            ], dim=1)  # [batch_size, small_batch_num, window_num, horizon, nodes_num]
        horizoned_data = horizoned_data.view(-1 , horizoned_data.shape[2], horizoned_data.shape[3], horizoned_data.shape[4])  # [batch_size*small_batch_num, window_num, horizon, nodes_num]

        weights_generation_module = models.GRUModule_small(input_dim=window_size*nodes_num, nodes_num=nodes_num, hidden_dim=nodes_num*nodes_num*hidden*(1+diffusion_size), window_size=window_size, hidden=hidden)
        weights_generation_module = weights_generation_module.to(device)

        ts = torch.linspace(0, (horizon-1)*1, horizon).to(device)

        funcs = nn.ModuleList([
            models.TimeHomogeneousSDE(
                dims=[nodes_num, diffusion_size],
                hidden=hidden,
                device=device
            )
            for j in range(window_num)
        ])

        optimizer = torch.optim.Adam(
            weights_generation_module.parameters(),
            lr=lr
        )

        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        elif scheduler_name == 'LambdaLR':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(step**-0.5, step*(1000**-1.5)))
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
        losses=[]
        for i in range(epoch):
            
            loss = 0
            f_weights, g_weights = weights_generation_module(stacked_data)
            # f_weights: [batch_size, window_num, nodes_num, nodes_num, hidden]
            # g_weights: [batch_size, window_num, nodes_num, nodes_num, hidden, diffusion_size]
            f_weights_temp = torch.normal(mean=0, std=1, size=f_weights.size()).to(device)
            g_weights_temp = torch.normal(mean=0, std=1, size=g_weights.size()).to(device)

            sampled_f_weights = f_weights+causality_std*f_weights_temp
            sampled_g_weights = g_weights+causality_std*g_weights_temp

            for j in range(window_num):
                funcs[j].f_weight = sampled_f_weights[:, j, :, :, :].squeeze(0)
                funcs[j].g_weight = sampled_g_weights[:, j, :, :, :, :].squeeze(0)
                predicted_ys = sdeint(funcs[j], horizoned_data[:,j,0,:], ts, method="euler", dt=1)
                real_ys = horizoned_data[:,j,:,:]
                loss = loss + F.mse_loss(predicted_ys.permute(1,0,2), real_ys.detach(), reduction='mean')
                if reg != 0:
                    loss += reg*funcs[j].l2_reg()
            losses.append(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            for j in range(window_num):
                if threshold != 0:
                    models.proximal(funcs[j].f_weight, threshold=threshold, target=target)
            graphs = np.array([func.causal_graph(threshold = 0) for func in funcs])
            if i % 1 == 0 and show_graph:
                fig, axs = plt.subplots(1, 3, figsize=(10, 2.3))
                fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)
                axs[0].plot(ts[:horizon].detach().squeeze().cpu().numpy(), real_ys[0,:,:].detach().squeeze().cpu().numpy())
                axs[1].plot(predicted_ys[:,0,:].detach().squeeze().cpu())
                axs[1].set_title("Iteration = %i" % i + ",  " + "Loss = %1.3f" % loss)
                cax = axs[2].matshow(np.mean(np.mean(graphs, axis=3), axis=0), cmap='Blues')
                fig.colorbar(cax)
                if real_causality is not None:
                    auc = roc_auc_score(real_causality.flatten(), np.mean(np.mean(graphs, axis=3), axis=0).flatten())
                    axs[2].set_title("AUC = %1.8f" % auc)
                plt.show()
                clear_output(wait=True)



    return graphs, real_causality




if __name__ == '__main__':
    main()