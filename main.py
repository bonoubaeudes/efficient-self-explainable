import time
from math import trunc
import sys

from data.dataset import build_dataloaders
from models.bagnet import get_bagnet_model
from models.builder import builder
from models.resnet import get_resnet_model
from quantization.quantize import quantize
from train import fine_tune, evaluate
from utils.config import parse_args
import yaml

from utils.func import plot_losses, plot_conf_matrix
from utils.metrics import Metric, model_mb
from utils.visualize_bagnet_evidence import visualize_bagnet_evidence


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    epochs = args.epochs or cfg["training"]["epochs"]
    lr     = args.lr     or cfg["training"]["learning_rate"]
    device = cfg["training"]["fine_tune_device"]
    n_calib_batches = cfg["quantization"]["n_calib_batches"]
    save_path_resnet_fp32 = cfg["paths"]["resnet_fp32_checkpoint"]

    num_classes = cfg["model"]["num_classes"]
    since = time.time()
    train_loader, val_loader, test_loader, calib_loader, train_dataset, class_names, _ = build_dataloaders(
        data_dir    = cfg["data"]["data_dir"],
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        val_split   = cfg["data"]["val_split"],
        test_split  = cfg["data"]["test_split"],
        random_state= cfg["data"]["random_state"],
        image_size  = cfg["data"]["image_size"],
        crop_size   = cfg["data"]["crop_size"],
    )

    model, checkpoint = builder(cfg, num_classes)
    metric_calculator = Metric(cfg)
    print('h')
    train_losses, val_losses = fine_tune(cfg, train_dataset, model, train_loader, val_loader, metric_calculator, epochs, checkpoint)
    evaluate(cfg, model, checkpoint, test_loader, metric_calculator, type_ds='test')
    time_elapsed = time.time() - since
    print('Training and evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model_quantize = quantize(cfg, model, calib_loader, n_calib_batches)
    print(model_mb((model_quantize)))
    #plot_losses(train_losses, val_losses)
    #plot_conf_matrix(metric_calculator, class_names)
    image_path="MLB/1621319173100.jpg"
    #visualize_bagnet_evidence(cfg, image_path, model)


if __name__ == '__main__':
    main()