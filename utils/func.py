import os
from typing import Optional, List

import yaml
import torch
import shutil
import socket
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from datetime import datetime

from matplotlib import gridspec
from tqdm import tqdm
from munch import munchify
from torch.utils.data import DataLoader

from utils.const import regression_loss
from utils.metrics import Metric

CIN_hostname = []

def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        '-config',
        type=str,
        default='./configs/default.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-paths',
        type=str,
        default='./configs/paths.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(
        '-overwrite',
        action='store_true',
        default=False,
        help='Overwrite file in the save path.'
    )
    parser.add_argument(
        '-print_config',
        action='store_true',
        default=False,
        help='Print details of configs.'
    )
    args = parser.parse_args()
    return args

def load_save_paths(cfg):
    timestamp_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")  
    save_path_model = 'Outputs/tmp/BagNet_SA' if cfg.base.test else f'Outputs/BagNet_SA/{cfg.base.dataset}'
    save_path = os.path.join(os.path.expanduser('~'), save_path_model, timestamp_str) 
    return save_path

def data_path(cfg, cfg_path):
    dset = cfg.base.dataset
    
    cfg.data.std = cfg_path[dset].std 
    cfg.data.mean = cfg_path[dset].mean
    
    cfg.dset['val_csv'] = cfg_path[dset].val_csv
    cfg.dset['test_csv'] = cfg_path[dset].test_csv
    cfg.dset['train_csv'] = cfg_path[dset].train_csv      

    cfg.data['fname'] = cfg_path[dset].fname 
    cfg.data['binary'] = cfg_path[dset].binary
    cfg.data['target'] = cfg_path[dset].target
    cfg.data['threshold'] = cfg_path[dset].threshold
    cfg.data['input_size'] = cfg_path[dset].input_size    
    cfg.data['num_classes'] = cfg_path[dset].num_classes 
    
    cfg.dset['root'] = cfg_path[dset].root
    cfg.dset['data_dir'] = cfg_path[dset].data_dir
    return cfg

def load_config(args):
    with open(args.config, 'r') as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    with open(args.paths, 'r') as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    return munchify(cfg), munchify(paths) 

def load_conf_file(config_file_path):
    '''
        Load the conf file containing all the parameters with the dataset paths

        input:
            - config_file_path (str): path to the configuration file

        output
            - a dictionary containing the conf parameters
    '''
    with open(config_file_path) as fhandle:
        cfg = yaml.safe_load(fhandle)
        
    cfg = munchify(cfg)
    return cfg


def copy_config(src, dst):
    shutil.copy(src, dst)


### -----------------------------
def copy_file(folds, dest_dir):
    # copy files: can be more customize
    for directory in folds:
        src_dir = os.path.join(os.getcwd(), directory)
        dst_dir = os.path.join(dest_dir, directory)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        files = os.listdir(src_dir)
        for file in files:
            fpath = os.path.join(src_dir, file)
            if file.startswith("__"):
                continue
            elif os.path.isdir(fpath):
                fs = os.listdir(fpath)
                for f in fs:
                    if not f.startswith(tuple(["__", "."])):
                        if os.path.isdir(os.path.join(fpath, f)):
                            continue ## norms and aother
                        shutil.copy2(os.path.join(fpath, f), dst_dir)
            else:
                shutil.copy2(fpath, dst_dir)
    
    #shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=dest_dir)    
    shutil.copy(src=os.path.join(os.getcwd(), 'train.py'), dst=dest_dir)    
    shutil.copy(src=os.path.join(os.getcwd(), 'main.py'), dst=dest_dir)


def save_config(config, path):
    with open(path, 'w') as file:
        yaml.safe_dump(config, file)


def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    mean, std = channel_mean.tolist(), channel_std.tolist()
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    return mean, std


def save_weights(model, save_path, mod=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()    
    torch.save(state_dict, save_path)
    
    if mod:
        torch.save(obj=model, f=save_path)


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


def print_config(configs):
    for name, config in configs.items():
        print('====={}====='.format(name))
        _print_config(config)
        print('=' * (len(name) + 10))
        print()


def _print_config(config, indentation=''):
    for key, value in config.items():
        if isinstance(value, dict):
            print('{}{}:'.format(indentation, key))
            _print_config(value, indentation + '    ')
        else:
            print('{}{}: {}'.format(indentation, key, value))


def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets
    print('=========================')
    print('Dataset Loaded.')
    print('Categories:\t{}'.format(len(train_dataset.classes)))
    print('Training:\t{}'.format(len(train_dataset)))
    print('Validation:\t{}'.format(len(val_dataset)))
    print('Test:\t\t{}'.format(len(test_dataset)))
    print('=========================')


# unnormalize image for visualization
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# convert labels to onehot
def one_hot(labels, num_classes, device, dtype):
    y = torch.eye(num_classes, device=device, dtype=dtype)
    return y[labels]

# convert type of target according to criterion
def select_target_type(y, criterion):
    if criterion in ['cross_entropy', 'kappa_loss']:
        y = y.long()
    elif criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:
        y = y.float()
    elif criterion in ['focal_loss']:
        y = y.to(dtype=torch.int64)
    else:
        raise NotImplementedError('Not implemented criterion.')
    return y


# convert output dimension of network according to criterion
def select_out_features(num_classes, criterion):
    out_features = num_classes
    if criterion in regression_loss:
        out_features = 1
    return out_features


def matplotlib_roccurve(fpr_tpr_tuples, labels, points=None, point_labels=None):    
    """helper function to plot a roc curve
        fpr_tpr_tuples: list of tuples [(fpr, tpr), (fpr, tpr)]
    """
    
    if not len(fpr_tpr_tuples) == len(labels):
        raise ValueError('both inputs must have same length.')
        
    if points is not None:
        if not len(points) == len(point_labels):
            raise ValueError('both inputs must have same length.')
        
        
    fig, ax = plt.subplots()
    
    for ii in range(len(fpr_tpr_tuples)):
        plt.plot(fpr_tpr_tuples[ii][0], fpr_tpr_tuples[ii][1], label=labels[ii])
        
    if points is not None:     
        for ii in range(len(points)):
            plt.plot(points[ii][0], points[ii][1], 'o', label=point_labels[ii])
            
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")
    ax.set_aspect('equal')

    return fig

def matplotlib_prec_recall_curve(precison_recall_tuples, labels):    
    """helper function to plot a roc curve
        fpr_tpr_tuples: list of tuples [(fpr, tpr), (fpr, tpr)]
    """
    
    if not len(precison_recall_tuples) == len(labels):
        raise ValueError('both inputs must have same length.')        
        
    fig, ax = plt.subplots()
    
    for ii in range(len(precison_recall_tuples)):
        plt.plot(precison_recall_tuples[ii][0], precison_recall_tuples[ii][1], label=labels[ii])
            
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    ax.set_aspect('equal')

    return fig

def plot_conf_matrix(metric_calculator: Metric, class_names):
    """  combine the confusion matrix with the approproate labels to make it easier to visualize """

    cm = metric_calculator.get_auc_auprc(5)

    table = pd.DataFrame(
        cm,
        columns=class_names,
        index=class_names
    )

    # plot
    plt.figure(figsize=(6,5))

    ax = sns.heatmap(
        table,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('True', fontsize=11)
    plt.title(f' — Accuracy: ', fontsize=12)

    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses, title="Training vs Validation Loss"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss", marker="o")
    plt.plot(val_losses,   label="Val loss",   marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()


def plot_all_sensitivity_curves(
        results_resnet,           # dict {'FP32': ..., 'FP16': ..., 'INT8': ...}
        results_dense_bagnet,     # dict {'FP32': ..., 'FP16': ..., 'INT8': ...}
        results_sparse_bagnet,    # dict {'FP32': ..., 'FP16': ..., 'INT8': ...}
        figsize=(11, 8),
        tick_labelsize=12,
        axis_labelsize=12,
        title_fontsize=13,
        suptitle_fontsize=15,
        save_path=None,
):
    """
    Trace les courbes de sensibilité pour :
      - BagNet33          (pleine largeur, haut)
      - Dense BagNet      (bas gauche)
      - Sparse BagNet     (bas droite)

    Chaque `results_*` est un dict dont les valeurs contiennent :
      - results[method]['mean']  : array (k_max+1,)
      - results[method]['lower'] : array (k_max+1,)   ← borne basse IC
      - results[method]['upper'] : array (k_max+1,)   ← borne haute IC
      - results[method]['ks']    : array (k_max+1,)   ← valeurs de k
    """

    styles = {
        'FP32': dict(color='#2176AE', ls='-',  marker='o', ms=5, label='FP32'),
        'FP16': dict(color='#D85A30', ls='--', marker='s', ms=5, label='FP16'),
        'INT8': dict(color='#3B6D11', ls=':',  marker='^', ms=5, label='INT8'),
    }

    panels = [
        ('ResNet-50',      results_resnet,        (0, slice(None, None))),
        ('dense BagNet',  results_dense_bagnet,  (1, 0)),
        ('sparse BagNet', results_sparse_bagnet, (1, 1)),
    ]

    fig = plt.figure(figsize=figsize, facecolor='white')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

    for title, results, gs_idx in panels:
        ax = fig.add_subplot(gs[gs_idx])

        for method, s in styles.items():
            if method not in results:
                continue
            r  = results[method]
            ks  = np.array(r['k'])
            m   = np.array(r['mean_score'])
            std = np.array(r['std_score'])
            lo  = m - std
            hi  = m + std

            ax.fill_between(ks, lo, hi, color=s['color'], alpha=0.15)
            ax.plot(ks, m,
                    color=s['color'], ls=s['ls'],
                    marker=s['marker'], markersize=s['ms'],
                    linewidth=2, label=s['label'])

        ax.set_title(f'Predictive Score vs k — {title}',
                     fontsize=title_fontsize, fontweight='bold', pad=7)
        ax.set_xlabel('Number of Masked Regions (k)', fontsize=axis_labelsize)
        ax.set_ylabel('Mean Predictive Class Score',  fontsize=axis_labelsize)
        ax.tick_params(labelsize=tick_labelsize)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0.2)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=tick_labelsize - 1, framealpha=0.7, loc='upper right')


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', facecolor='white')
        print(f"Figure saved : {save_path}")

    plt.show()

def plot_quantization_results(
        all_results: dict[str, dict],
        model_names: Optional[List[str]] = None,
        save_path: str = '/kaggle/working/quantization_results.png'
):
    """
    Parameters
    ----------
    all_results : dict
        Keys are model names (e.g. 'Dense BagNet', 'Sparse BagNet', 'ResNet').
        Each value is a dict with keys: 'FP32', 'FP16', 'INT8', each containing
        {'acc', 'size', 'pc_mean', 'pc_std', 'edge_mean', 'edge_std'}.
    model_names : list[str] | None
        Display order for models. Defaults to all_results.keys().
    save_path : str
        Output path.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if model_names is None:
        model_names = list(all_results.keys())

    n_models = len(model_names)
    color_bar  = '#C8722A'
    color_size = '#5C1A00'
    color_pc   = '#90EE90'
    color_edge = '#ADD8E6'
    width_bar  = 0.5
    w2         = 0.35

    fig, axes = plt.subplots(
        2, n_models,
        figsize=(3.5 * n_models, 6),
        constrained_layout=True
    )
    # axes[0, i] → accuracy + size
    # axes[1, i] → inference time

    for col, name in enumerate(model_names):
        res    = all_results[name]
        labels = list(res.keys())           # ['FP32', 'FP16', 'INT8']
        x      = np.arange(len(labels))

        accs   = [res[k]['acc']       for k in labels]
        sizes  = [res[k]['size']      for k in labels]
        pc_m   = [res[k]['pc_mean']   for k in labels]
        pc_s   = [res[k]['pc_std']    for k in labels]
        edge_m = [res[k]['edge_mean'] for k in labels]
        edge_s = [res[k]['edge_std']  for k in labels]

        # ── Row 0 : Accuracy + Model size ──────────────────────────
        ax1 = axes[0, col]
        bars = ax1.bar(x, accs, width=width_bar, color=color_bar, zorder=2)
        ax1.set_ylim(85, 100)
        ax1.set_ylabel('Accuracy (%)' if col == 0 else '', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=15)
        ax1.set_xlabel('Quantization variant', fontsize=14)
        ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)
        ax1.set_title(f'(a{col+1}) {name}', fontsize=11)

        for bar, acc in zip(bars, accs):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f'{acc:.1f}', ha='center', va='bottom', fontsize=11
            )

        ax1b = ax1.twinx()
        ax1b.plot(x, sizes, 'o-', color=color_size, linewidth=2, markersize=6, zorder=3)
        for xi, si in zip(x, sizes):
            ax1b.annotate(
                f'{si}', (xi, si),
                textcoords='offset points', xytext=(5, 4),
                fontsize=13, color=color_size
            )
        ax1b.set_ylabel('Model size (MB)' if col == n_models - 1 else '',
                        fontsize=14, color=color_size)
        ax1b.set_ylim(0, max(sizes) * 1.4)
        ax1b.tick_params(axis='y', labelcolor=color_size)

        if col == 0:
            ax1.legend(handles=[
                Patch(facecolor=color_bar, label='Accuracy (%)'),
                Line2D([0], [0], color=color_size, marker='o', label='Model size (MB)')
            ], fontsize=11, loc='lower left')

        # ── Row 1 : Inference time ──────────────────────────────────
        ax2 = axes[1, col]

        bars_pc   = ax2.bar(x - w2/2, pc_m,   w2, yerr=pc_s,   color=color_pc,
                            capsize=4, label='PC inference',   zorder=2)
        bars_edge = ax2.bar(x + w2/2, edge_m, w2, yerr=edge_s, color=color_edge,
                            capsize=4, label='Edge inference', zorder=2)
        ax2.bar_label(bars_pc,   fmt='%.0f', fontsize=12)
        ax2.bar_label(bars_edge, fmt='%.0f', fontsize=12)

        ax2.set_ylabel('Inference time (ms)' if col == 0 else '', fontsize=14)
        ax2.set_xlabel('Quantization variant', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=15)
        y_upper = max(edge_m) * 1.35
        ax2.set_ylim(0, y_upper)
        ax2.set_yticks(np.arange(0, y_upper + 50, 50))
        ax2.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)
        ax2.set_title(f'(b{col+1}) {name}', fontsize=11)

        if col == 0:
            ax2.legend(fontsize=9, loc='upper left')


    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {save_path}")
