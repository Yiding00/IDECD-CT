def main(config=None):
    from Modules.LoadData.load_data_starplus import get_dataloader, get_timelength_list, get_ROI_names
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
    import datetime
    import json
    from scipy.io import savemat
    import os
    
    default_config = {
        'hidden': 6,
        'causality_std': 1e-4,
        'horizon': 5,
        'stride': 1,
        'diffusion_size': 1,
        'epoch': 4000,
        'lr': 8e-5,
        'reg': 0,
        'target': 0,
        'threshold': 0,
        'show_graph': False,
        'device': 'cuda:1',
        'scheduler': 'CosineAnnealingLR',
        'delta': 1,
        'training': True
    }
    if config:
        default_config.update(config)
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
    scheduler_name=default_config['scheduler']
    delta=default_config['delta']
    training=default_config['training']
    data_loader = get_dataloader(batch_size = 1)
    timelength_list = get_timelength_list()
    changepoint_list = [0]
    temp = 0
    window_num = len(timelength_list)
    window_num = 5
    
    for i in range(window_num):
        temp += timelength_list[i]
        changepoint_list.append(temp)
    changepoint_list = np.array(changepoint_list)
    
    real_causality = None
    nodes_num = 25

    weights_generation_module = models.GRUModule(input_dim=nodes_num, nodes_num=nodes_num, hidden_dim=nodes_num*nodes_num*hidden*(1+diffusion_size), hidden=hidden, changepoint_list = changepoint_list)
    weights_generation_module = weights_generation_module.to(device)

    ts = torch.linspace(0, (horizon-1)*delta, horizon).to(device)

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

    if not training:
        weights_generation_module.load_state_dict(torch.load(config['path']+'_model_weights.pth'))
        ROI_names = get_ROI_names()
        for data, id, timelength_list in data_loader:
            data = data[:, :changepoint_list[window_num], :]
            loss = 0
            # starplus数据集中所有数据的网格划分方式相同
            data = data.float().to(device)  # [batch_size, time_length, nodes_num]
            batch_size, time_length, nodes_num = data.shape
            assert nodes_num == 25


            horizoned_data = []

            for i in range(window_num):
                len_temp = changepoint_list[i+1] - changepoint_list[i]
                temp = data[:, changepoint_list[i]:changepoint_list[i+1], :]
                assert len_temp >= horizon
                horizoned_data.append(torch.stack([
                    data[:, k*stride:k*stride+horizon, :].squeeze(0)
                    for k in range((len_temp-horizon)//stride+1)
                    ], dim=0))
            
            
            f_weights, g_weights = weights_generation_module(data)
            # f_weights: [batch_size, window_num, nodes_num, nodes_num, hidden]
            # g_weights: [batch_size, window_num, nodes_num, nodes_num, hidden, diffusion_size]
            f_weights_temp = torch.normal(mean=0, std=1, size=f_weights.size()).to(device)
            g_weights_temp = torch.normal(mean=0, std=1, size=g_weights.size()).to(device)

            sampled_f_weights = f_weights+causality_std*f_weights_temp
            sampled_g_weights = g_weights+causality_std*g_weights_temp

            for j in range(window_num):
                funcs[j].f_weight = sampled_f_weights[:, j, :, :, :].squeeze(0)
                funcs[j].g_weight = sampled_g_weights[:, j, :, :, :, :].squeeze(0)
                predicted_ys = sdeint(funcs[j], horizoned_data[j][:,0,:], ts, method="euler", dt=delta)

            graphs = np.array([func.causal_graph(threshold = 0) for func in funcs])
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            for i in range(window_num):
                ax = axs[i // 3, i % 3]
                ax.set_title(f"Window {i+1}", fontsize=25)
                mean_val = np.mean(graphs[i,:,:,:], axis=2)
                vmax = np.max(np.mean(graphs, axis=2))
                im = ax.matshow(mean_val.squeeze(), cmap='Blues', vmin=0, vmax=vmax)  # 去除单维度
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.ax.tick_params(labelsize=16)
            ax = axs[1, 2]
            ax.set_title(f"Average", fontsize=25)
            mean_val = np.mean(np.mean(graphs, axis=3), axis=0)
            im = ax.matshow(mean_val.squeeze(), cmap='Blues', vmin=0, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=16)
            plt.tight_layout()
            plt.savefig('Figs/Starplus/Starplus_'+str(id[0])+'_dECN.pdf')
            plt.show()
            clear_output(wait=True)

            fig, axs = plt.subplots(1, 3, figsize=(15,5))
            for i in range(1,window_num-1):
                ax = axs[i - 1]
                ax.set_title(f"Window {i+1} To Window {i+2}", fontsize=25)
                mean_val_i = np.mean(graphs[i,:,:,:], axis=2)
                mean_val_i1 = np.mean(graphs[i+1,:,:,:], axis=2)
                vmax = np.max(np.mean(graphs[2:,:,:,:]-graphs[1:-1,:,:,:], axis=2))
                vmin = np.min(np.mean(graphs[2:,:,:,:]-graphs[1:-1,:,:,:], axis=2))
                vm = max(abs(vmin), abs(vmax))
                im = ax.matshow(mean_val_i1 - mean_val_i.squeeze(), cmap='coolwarm', vmin=-vm, vmax=vm)  # 去除单维度
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.ax.tick_params(labelsize=16)

                # 计算正负矩阵
                diff = mean_val_i1-mean_val_i
                positive_matrix = np.zeros_like(diff)
                negative_matrix = np.zeros_like(diff)
                threshold = 0.02
                positive_matrix[diff > threshold] = diff[diff > threshold]
                negative_matrix[diff < -threshold] = diff[diff < -threshold]
                negative_matrix = -negative_matrix
                savemat('Figs/Starplus/Starplus_'+str(id[0])+f'positive_matrix{i+2}.mat', {'positive_matrix': positive_matrix})
                savemat('Figs/Starplus/Starplus_'+str(id[0])+f'negative_matrix{i+2}.mat', {'negative_matrix': negative_matrix})

                # 计算diff矩阵中最大的10个值及其坐标+最有影响力的ROI
                flat_indices = np.argpartition(np.abs(diff).flatten(), -10)[-10:]
                coordinates = np.unravel_index(flat_indices, diff.shape)
                top_values = diff.flatten()[flat_indices]
                sorted_indices = np.argsort(-np.abs(top_values))
                top_values = top_values[sorted_indices]
                coordinates = tuple(np.array(coords)[sorted_indices] for coords in coordinates)

                row_sum = np.sum(np.abs(diff), axis=1)
                col_sum = np.sum(np.abs(diff), axis=0)
                total_influence = row_sum + col_sum
                top10_indices = np.argsort(-total_influence)[:10]  # 降序排序后取前10
                top10_values = total_influence[top10_indices]

                with open('Figs/Starplus/Starplus_'+str(id[0])+f'top_values{i+2}.txt', "w", encoding="utf-8") as f:
                    f.write("矩阵中最大的10个值及其坐标：\n")
                    f.write("格式：值 (行, 列)\n")
                    f.write("=" * 30 + "\n")
                    for i in range(10):
                        f.write(f"{i+1}. 值: {top_values[i]:.2f}, 坐标: {ROI_names[coordinates[0][i]]}, {ROI_names[coordinates[1][i]]}\n")
                    f.write("\n最有影响力的ROI：\n")
                    f.write("格式：值 (ROI)\n")
                    f.write("=" * 30 + "\n")
                    for j in range(10):
                        f.write(f"{j+1}. 值: {top10_values[j]:.2f}, 坐标: {ROI_names[top10_indices[j]]}\n")

            plt.tight_layout()
            plt.savefig('Figs/Starplus/Starplus_'+str(id[0])+'_dECN_diff.pdf')
            plt.show()
            clear_output(wait=True)
            

    if training:
        for k in range(epoch):
            loss_i = 0
            for data, id, timelength_list in data_loader:
                data = data[:, :changepoint_list[window_num], :]
                loss = 0
                # starplus数据集中所有数据的网格划分方式相同
                data = data.float().to(device)  # [batch_size, time_length, nodes_num]
                batch_size, time_length, nodes_num = data.shape
                assert nodes_num == 25


                horizoned_data = []

                for i in range(window_num):
                    len_temp = changepoint_list[i+1] - changepoint_list[i]
                    temp = data[:, changepoint_list[i]:changepoint_list[i+1], :]
                    assert len_temp >= horizon
                    horizoned_data.append(torch.stack([
                        data[:, k*stride:k*stride+horizon, :].squeeze(0)
                        # data[:, k*stride:k*stride+horizon, :]
                        for k in range((len_temp-horizon)//stride+1)
                        ], dim=0))
                
                
                f_weights, g_weights = weights_generation_module(data)
                # f_weights: [batch_size, window_num, nodes_num, nodes_num, hidden]
                # g_weights: [batch_size, window_num, nodes_num, nodes_num, hidden, diffusion_size]
                f_weights_temp = torch.normal(mean=0, std=1, size=f_weights.size()).to(device)
                g_weights_temp = torch.normal(mean=0, std=1, size=g_weights.size()).to(device)

                sampled_f_weights = f_weights+causality_std*f_weights_temp
                sampled_g_weights = g_weights+causality_std*g_weights_temp

                for j in range(window_num):
                    funcs[j].f_weight = sampled_f_weights[:, j, :, :, :].squeeze(0)
                    funcs[j].g_weight = sampled_g_weights[:, j, :, :, :, :].squeeze(0)
                    # funcs[j].f_weight = sampled_f_weights[:, j, :, :, :]
                    # funcs[j].g_weight = sampled_g_weights[:, j, :, :, :, :]
                    predicted_ys = sdeint(funcs[j], horizoned_data[j][:,0,:], ts, method="euler", dt=delta)
                    real_ys = horizoned_data[j]
                    loss = loss + F.mse_loss(predicted_ys.permute(1,0,2), real_ys.detach(), reduction='mean')
                    if reg != 0:
                        loss += reg*funcs[j].l2_reg()
                loss_i += loss
                
            optimizer.zero_grad()
            loss_i.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            for j in range(window_num):
                if threshold != 0:
                    models.proximal(funcs[j].f_weight, threshold=threshold, target=target)
                    # proximal(funcs[j].g_weight, threshold=threshold, target=target)
            graphs = np.array([func.causal_graph(threshold = 0) for func in funcs])
            losses.append(loss_i.detach().cpu().numpy())
            if k % 1 == 0 and show_graph:
                fig, axs = plt.subplots(2, 3, figsize=(10, 5))
                fig.tight_layout(pad=0.2, w_pad=2, h_pad=3)
                axs[0,0].plot(ts[:horizon].detach().squeeze().cpu().numpy(), real_ys[0,:,:].detach().squeeze().cpu().numpy())
                axs[0,1].plot(predicted_ys[:,0,:].detach().squeeze().cpu())
                axs[0,1].set_title("Iteration = %i" % k + ",  " + "Loss = %1.3f" % loss)
                cax = axs[0,2].matshow(np.mean(np.mean(graphs, axis=3), axis=0), cmap='Blues')
                fig.colorbar(cax)
                if real_causality is not None:
                    auc = roc_auc_score(real_causality.flatten(), np.mean(np.mean(graphs, axis=3), axis=0).flatten())
                    axs[0,2].set_title("AUC = %1.8f" % auc)
                axs[1,0].plot(losses)
                axs[1,0].set_title("LR = %1.8f" % optimizer.param_groups[0]['lr'])
                if k > 10:
                    axs[1,1].plot(losses[k-10:])
                plt.show()
                clear_output(wait=True)
            # if i % 1000 == 0:
            #     print("Epoch = %i" % i + ",  " + "Loss = %1.3f" % loss)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(weights_generation_module.state_dict(), os.path.join('Evaluation_results', f'25-SDE_starplus_{timestamp}_model_weights.pth'))
        np.savetxt(os.path.join('Evaluation_results', f'25-SDE_starplus_{timestamp}_loss'), losses)
        with open(os.path.join('Evaluation_results', f'25-SDE_starplus_{timestamp}_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        # print(losses[-1])

    return graphs, losses

def reconstruct_second_window(horizoned_data, original_window_length, horizon, stride):
    import torch
    reconstructed_window = torch.zeros((original_window_length))
    for i in range(original_window_length):
        if i< horizon:
            reconstructed_window[i] = horizoned_data[0][i]
        else:
            reconstructed_window[i] = horizoned_data[i - horizon + 1][0]
    return reconstructed_window


if __name__ == '__main__':
    condig = {
    "hidden": 6,
    "lr": 8.208807435843559e-05,
    "reg": 0,
    "causality_std": 0.00016134947723961851,
    "horizon": 5,
    "scheduler": "CosineAnnealingLR",
    "epoch": 3264
}
    main(config)