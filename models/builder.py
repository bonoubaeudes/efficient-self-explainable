from models.bagnet import get_bagnet_model
from models.resnet import get_resnet_model


def builder(cfg, num_classes:int):
    save_path_resnet_fp32 = cfg["paths"]["resnet_fp32_checkpoint"]
    bagnet_fp32_checkpoint = cfg["paths"]["bagnet_fp32_checkpoint"]
    bagnet = True if "bagnet" in cfg["training"]["network"] else False
    if bagnet:
        return get_bagnet_model(num_classes = num_classes), bagnet_fp32_checkpoint
    else:
        return get_resnet_model(num_classes = num_classes), save_path_resnet_fp32

