import copy

import torch
import torch.nn as nn


def ptq_quantize_static_bagnet(model, calib_loader, backend='qnnpack', n_batches=50):
    model = copy.deepcopy(model).cpu()
    model.eval()

    modules_to_fuse = get_fusable_modules(model)

    model = fuse_bagnet_modules(model, modules_to_fuse)

    model.qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.HistogramObserver.with_args(
            quant_min=0, quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False
        ),
        weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(
            quant_min=-128, quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    )

    model.conv2.qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.HistogramObserver.with_args(
            quant_min=-128, quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False
        ),
        weight=torch.ao.quantization.PerChannelMinMaxObserver.with_args(
            quant_min=-128, quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric
        )
    )

    model_prepared = torch.ao.quantization.prepare(model)

    # Calibration
    model_calibrated = calibrate(model_prepared, calib_loader, n_batches)

    model_int8 = torch.ao.quantization.convert(model_calibrated)
    print("✓ Conversion INT8 terminée")

    return model_int8


def calibrate(model, dataloader, num_batches=100):
    model.eval()
    model.to('cpu')
    print(f"Calibration sur {num_batches} batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            model(images.to('cpu'))
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{num_batches} batches traités")
    print("✓ Calibration terminée")
    return model


def fuse_bagnet_modules(model, modules_to_fuse):
    model.eval()

    print(f"Groupes à fusionner : {len(modules_to_fuse)}")

    try:
        from torch.ao.quantization import fuse_modules
        fused = fuse_modules(model, modules_to_fuse, inplace=True)
        print("✅ Fusion réussie")
        return fused
    except Exception as e:
        print(f"❌ Fusion échouée : {e}")
        return model


def get_fusable_modules(model):

    modules_to_fuse = []

    for module_name, module in model.named_modules():

        children = list(module.named_children())
        child_names = [name for name, _ in children]
        child_modules = [m for _, m in children]

        i = 0

        while i < len(child_modules) - 1:

            # ---------------------------------------------------
            # Conv + BN + ReLU
            # ---------------------------------------------------
            if (
                    i + 2 < len(child_modules)
                    and isinstance(child_modules[i], nn.Conv2d)
                    and isinstance(child_modules[i + 1], nn.BatchNorm2d)
                    and isinstance(child_modules[i + 2], nn.ReLU)
            ):

                fuse_group = [
                    f"{module_name}.{child_names[i]}" if module_name else child_names[i],
                    f"{module_name}.{child_names[i+1]}" if module_name else child_names[i+1],
                    f"{module_name}.{child_names[i+2]}" if module_name else child_names[i+2],
                ]

                modules_to_fuse.append(fuse_group)

                i += 3
                continue

            # ---------------------------------------------------
            # Conv + BN
            # ---------------------------------------------------
            elif (
                    i + 1 < len(child_modules)
                    and isinstance(child_modules[i], nn.Conv2d)
                    and isinstance(child_modules[i + 1], nn.BatchNorm2d)
            ):

                fuse_group = [
                    f"{module_name}.{child_names[i]}" if module_name else child_names[i],
                    f"{module_name}.{child_names[i+1]}" if module_name else child_names[i+1],
                ]

                modules_to_fuse.append(fuse_group)

                i += 2
                continue

            i += 1

    return modules_to_fuse