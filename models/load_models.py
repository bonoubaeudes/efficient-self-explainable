import copy
import os

import torch
import torch.nn as nn
from torch.ao.quantization import QConfigMapping, MovingAveragePerChannelMinMaxObserver, MovingAverageMinMaxObserver, \
    QConfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from models.bagnet import get_bagnet_model
from models.resnet import get_resnet_model
from quantization.ptq_bagnet import get_fusable_modules, fuse_bagnet_modules


def load_models(cfg, num_classes):
    models = {
        "fp32bagnet": load_fp32bagnet_model(cfg, num_classes),
        "fp32resnet": load_fp32resnet_model(cfg, num_classes)
    }
    return models

def load_fp32bagnet_model(cfg, num_classes:int) -> nn.Module:
    path = cfg["paths"]["bagnet_fp32_checkpoint"]
    model = get_bagnet_model(num_classes)
    model.load_state_dict(torch.load(path))
    print("✓ FP32 Bagnet loaded")
    return model

def load_int8bagnet_model(cfg, num_classes:int) -> nn.Module:
    path = cfg["paths"]["bagnet_int8_checkpoint"]
    model = get_bagnet_model(num_classes)
    modules_to_fuse = get_fusable_modules(model)
    model = fuse_bagnet_modules(model, modules_to_fuse)
    model.qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.HistogramObserver.with_args(
            quant_min=0, quant_max=255, dtype=torch.quint8,
            qscheme=torch.per_tensor_affine, reduce_range=False),
        weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric)
    )
    model.conv2.qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.HistogramObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric, reduce_range=False),
        weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(
            quant_min=-128, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric)
    )
    torch.ao.quantization.prepare(model, inplace=True)
    torch.ao.quantization.convert(model, inplace=True)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval().cpu()
    print("✓ INT8 Bagnet loaded ")
    return model


def load_fp32resnet_model(cfg, num_classes:int) -> nn.Module:
    path = cfg["paths"]["resnet_fp32_checkpoint"]
    model = get_resnet_model(num_classes)
    model.load_state_dict(torch.load(path))
    print("✓ FP32 resnet loaded")
    return model

def load_int8resnet_model(cfg, num_classes:int) -> nn.Module:
    path = cfg["paths"]["resnet_int8_checkpoint"]
    model = get_resnet_model(num_classes)
    qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    example_input   = torch.randn(1, 3, 224, 224)
    prepared  = prepare_fx(model, qconfig_mapping, example_input)
    model = convert_fx(prepared)
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("✓ INT8 resnet loaded ")
    return model

def from_fp32_to_fp16(model:nn.Module) -> nn.Module:
    fp16model = copy.deepcopy(model).half()
    return fp16model