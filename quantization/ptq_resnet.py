import copy
import torch

from torch.ao.quantization import QConfig, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, \
    QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


def ptq_quantization_resnet(model, calib_loader, n_batches=50):
    model = copy.deepcopy(model).to("cpu").eval()

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

    print("Préparation du modèle pour calibration...")
    prepared = prepare_fx(model, qconfig_mapping, example_input)

    with torch.no_grad():
        imgs, lbls = next(iter(calib_loader))
        acc = (prepared(imgs).argmax(1) == lbls).float().mean().item() * 100
        print(f"   Sanity check prepare_fx : {acc:.1f}%")
        if acc < 50:
            raise RuntimeError(
                f"Modèle corrompu avant calibration (acc={acc:.1f}%). "
                "Vérifiez que fine_tune() retourne model.to('cpu').eval()."
            )

    print(f"Calibration en cours sur {n_batches} batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            prepared(images)
            if i + 1 >= n_batches:
                break

    quantized = convert_fx(prepared)

    with torch.no_grad():
        imgs, lbls = next(iter(calib_loader))
        acc = (quantized(imgs).argmax(1) == lbls).float().mean().item() * 100
        print(f"   Sanity check convert_fx  : {acc:.1f}%")

    print("✅ Quantification INT8 terminée.")
    return quantized