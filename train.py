from email.policy import strict
from tabnanny import check

import torch
import torch.nn as nn
from torch.cuda import device
from torch.utils.data import DataLoader

from utils.func import save_weights, select_target_type
from utils.loss import initialize_loss
from utils.metrics import Metric
from utils.optimizer import initialize_optimizer
from utils.scheduler import ScheduledWeightedSampler


def fine_tune(
        cfg,
        train_dataset,
        model: nn.Module,
        train_loader,
        val_loader,
        metric_calculator,
        epochs: int = 10,
        save_path: str = "results/best_fp32resnet_model.pth"
) :
    weighted_sampler = initialize_sampler(cfg, train_dataset)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        avg_train_loss, train_acc = train(cfg, model, loss_function, train_loader, metric_calculator)
        train_losses.append(avg_train_loss)
        avg_val_loss, val_acc = validation(cfg, model, val_loader, loss_function)
        val_losses.append(avg_val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_weights(model, save_path)
            print("  ✅ Best model saved.")
        print(f"Epoch {epoch+1}/{epochs}"
              f"  train_loss={train_losses[-1]:.4f}"
              f"  train_acc={train_acc:.4f}"
              f"  val_loss={avg_val_loss:.4f}"
              f"  val_acc={val_acc:.4f}")
    return train_losses, val_losses

def l1_regularization(activation_maps, lambda_l1=1e-6):
    l1_norm = torch.abs(activation_maps).sum()
    return lambda_l1 * l1_norm

def train(cfg, model, loss_function, train_loader, metric_calculator):
    device = cfg["training"]["fine_tune_device"]
    optimizer = initialize_optimizer(cfg, model)

    model.train()
    bagnet = True if 'bagnet' in cfg["training"]["network"] else False
    total_loss = train_correct = train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if bagnet:
            logits, evidence_maps, _ = model(images)
            loss = loss_function(logits, labels) + l1_regularization(evidence_maps)
        else:
            logits = model(images)
            loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        train_correct += (logits.argmax(1) == labels).sum().item()
        train_total   += labels.size(0)
    train_acc = train_correct/train_total
    return total_loss / len(train_loader),  train_acc


def validation(cfg, model, val_loader, loss_function):
    device = cfg["training"]["fine_tune_device"]
    model.eval()
    val_loss = val_correct = val_total = 0
    bagnet = True if 'bagnet' in cfg["training"]["network"] else False
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            if bagnet:
                logits, evidence_maps,_   = model(images)
                val_loss    += loss_function(logits, labels).item() + l1_regularization(evidence_maps)
            else:
                logits   = model(images)
                val_loss    += loss_function(logits, labels).item()
            val_correct += (logits.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)
    val_acc = val_correct/val_total
    return val_loss / len(val_loader), val_acc

def evaluate(cfg, model, checkpoint, test_loader, estimator, type_ds):
    weights =torch.load(checkpoint)
    loss = nn.CrossEntropyLoss()

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(weights, strict=True)
    else:
        model.load_state_dict(weights, strict=True)

    eval(cfg, model, test_loader, estimator, loss_func=loss)
    if cfg["data"]["binary"]:
        list_auc, list_auprc, list_others = estimator.get_auc_auprc(5)
        auc = list_auc[0]
        auprc, sens, prec, spec = list_auprc[0], list_others[0], list_others[1], list_others[2]

        print('Finished! {} acc {}'.format(type_ds, estimator.get_accuracy(6)[0]))
        print('loss:', estimator.get_val_loss())
        print('AUC: {}, sens: {}, spec: {}, prec: {}, AUPRC: {}'.format(auc, sens, spec, prec, auprc))
        print('Confusion Matrix:')
        print(estimator.conf_mat)
    else:
        print('acc.: {}'.format(estimator.get_accuracy(6)[0]))
        print('binary acc.: {}'.format(estimator.get_accuracy(6)[1]))
        print('kappa: {}'.format(estimator.get_kappa(6)))
        print(estimator.conf_mat)



def eval(cfg, model, dataloader, metric_calculator:Metric, loss_func=None):
    model.eval()
    device = cfg["training"]["fine_tune_device"]
    criterion = cfg["training"]["criterion"]
    torch.set_grad_enabled(False)
    l = {}

    if loss_func:
        loss_function = loss_func
        l['op'] = lambda tensor: tensor.item()
    else:
        loss_function = lambda a,b:1
        l['op'] = lambda a:0

    metric_calculator.reset()
    epoch_loss, avg_val_loss = 0, 0
    for step, test_data in enumerate(dataloader):
        images, labels = test_data
        images, labels = images.to(device), labels.to(device)
        labels = select_target_type(labels, criterion)
        if 'bagnet' in cfg["training"]["network"]:
            pred, evidence_maps, _ = model(images)
            loss = loss_function(pred, labels) + l1_regularization(evidence_maps)
        else:
            pred = model(images)
            loss = loss_function(pred, labels)
        metric_calculator.update(pred, labels)
        epoch_loss += l['op'](loss)
        avg_val_loss = epoch_loss / (step + 1)
    if loss_func:
        metric_calculator.update_val_loss(avg_val_loss)

    model.train()
    torch.set_grad_enabled(True)



def initialize_sampler(cfg, train_dataset):
    sampling_strategy = cfg["data"]["sampling_strategy"]
    if sampling_strategy == 'class_balanced':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, 1)
    elif sampling_strategy == 'progressively_balanced':
        weighted_sampler = ScheduledWeightedSampler(train_dataset, cfg.data.sampling_weights_decay_rate)
    else:
        weighted_sampler = None
    return weighted_sampler

