import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr",     type=float, default=None)
    return parser.parse_args()
