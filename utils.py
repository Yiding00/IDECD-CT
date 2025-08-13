import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve,f1_score, auc, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prfs
import torch
import random
import json
import pandas as pd
from datetime import datetime
import importlib

def get_model(TargetBaseline, TargetData):
    Baseline_name = f"Baselines.{TargetBaseline}.run_{TargetData}"
    load_data_module = importlib.import_module(Baseline_name)
    importlib.reload(load_data_module)
    Baseline_model = getattr(load_data_module, "main")
    return Baseline_model


def save_evaluation_json(all_evaluations, TargetBaseline, TargetData, config, graphs):
    df = pd.DataFrame(all_evaluations)
    mean_results = df.mean()
    std_results = df.std()
    confident_interval = std_results * 1.96
    mean_results['description'] = 'mean'
    std_results['description'] = 'std'
    confident_interval['description'] = 'ci'
    all_evaluations.append(mean_results.to_dict())
    all_evaluations.append(std_results.to_dict())
    all_evaluations.append(confident_interval.to_dict())
    if config is not None:
        all_evaluations.append(config) 
    filename = f"Evaluation_results/{TargetBaseline}_{TargetData}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_evaluations.json"
    filename1 = f"Evaluation_results/{TargetBaseline}_{TargetData}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_evaluations.npy"
    # 将结果写入JSON文件
    with open(filename, 'w') as f:
        json.dump(all_evaluations, f, indent=4, cls=NumpyEncoder)  # 处理numpy数据类型的编码
    np.save(filename1, graphs)
    print(f"评估结果已保存至 {filename}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def set_seed(seed, det= True):
    # Python内置随机模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # cuDNN相关设置（确保确定性计算）
    if det:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 对于DataLoader的workers
    torch.backends.cudnn.enabled = False  # 对于某些旧版本可能需要


def evaluate_result(causality_true, causality_pred, threshold=None):
    # max_row = causality_pred.max(axis=1)
    # causality_pred = causality_pred / (np.repeat(max_row[:, np.newaxis], max_row.shape[0], 1) + 1e-6)
    causality_pred[causality_pred > 1] = 1
    causality_true = np.abs(causality_true).flatten()
    causality_pred = np.abs(causality_pred).flatten()
    if threshold is None:
        threshold = np.percentile(causality_pred, 80)
    roc_auc = roc_auc_score(causality_true, causality_pred)
    fpr, tpr, _ = roc_curve(causality_true, causality_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(causality_true, causality_pred)
    pr_auc = auc(recall_curve, precision_curve)
    causality_pred[causality_pred > threshold] = 1
    causality_pred[causality_pred <= threshold] = 0
    precision, recall, F1, _ = prfs(causality_true, causality_pred)
    accuracy = accuracy_score(causality_true, causality_pred)
    shd = np.sum(causality_true != causality_pred)
    cs = calculate_cs(causality_true, causality_pred)
    evaluation = {'accuracy': accuracy, 'precision': precision[1], 'recall': recall[1], 'F1': F1[1],
                  'ROC_AUC': roc_auc, 'PR_AUC': pr_auc, 'SHD': shd, 'CS': cs}
    plot = {'FPR': fpr, 'TPR': tpr, 'PC': precision_curve, 'RC': recall_curve}
    return evaluation, plot

def calculate_cs(true_adj_matrix, prob_matrix):
    """
    计算两个n*n矩阵的余弦相似度
    
    参数：
    true_adj_matrix, prob_matrix -- 输入的两个n*n矩阵（列表或numpy数组）
    
    返回：
    余弦相似度值，范围[-1,1]
    """
    vec_c = np.array(true_adj_matrix).flatten()
    vec_a = np.array(prob_matrix).flatten()
    
    # 处理全零矩阵的情况
    norm_c = np.linalg.norm(vec_c)
    norm_a = np.linalg.norm(vec_a)
    if norm_c == 0 or norm_a == 0:
        return 0.0
    
    return np.dot(vec_c, vec_a) / (norm_c * norm_a)