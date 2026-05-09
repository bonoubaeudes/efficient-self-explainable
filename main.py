from math import trunc

from data.dataset import build_dataloaders
from models.resnet import get_model
from train import fine_tune, evaluate
from utils.config import parse_args
import yaml

from utils.func import plot_losses
from utils.metrics import Metric


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    epochs = args.epochs or cfg["training"]["epochs"]
    lr     = args.lr     or cfg["training"]["learning_rate"]
    device = cfg["training"]["fine_tune_device"]
    save_path_resnet_fp32 = cfg["paths"]["resnet_fp32_checkpoint"]

    num_classes = cfg["model"]["num_classes"]
    train_loader, val_loader, test_loader, _, train_dataset, class_names, _ = build_dataloaders(
        data_dir    = cfg["data"]["data_dir"],
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        val_split   = cfg["data"]["val_split"],
        test_split  = cfg["data"]["test_split"],
        random_state= cfg["data"]["random_state"],
        image_size  = cfg["data"]["image_size"],
        crop_size   = cfg["data"]["crop_size"],
    )
    model = get_model(num_classes)
    metric_calculator = Metric(cfg)
    checkpoint = save_path_resnet_fp32
    train_losses, val_losses = fine_tune(cfg, train_dataset, model, train_loader, val_loader, metric_calculator, epochs)
    evaluate(cfg, model, checkpoint, test_loader, metric_calculator, type_ds='test')
    plot_losses(train_losses, val_losses)


if __name__ == '__main__':
    main()