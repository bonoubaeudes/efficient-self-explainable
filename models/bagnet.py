import torch
import torch.nn as nn
from bagnets.pytorchnet import bagnet33
import bagnets.pytorchnet as bpn

class Bagnet(nn.Module):
    def __init__(self, model, num_classes):
        super(Bagnet, self).__init__()
        last_block = list(model.layer4.children())[-1]
        num_channlast_block    = list(model.layer4.children())[-1]
        num_channels  = last_block.conv3.out_channels       # 2048 pour ResNet50

        self.sequential = nn.Sequential(*list(model.children())[:-2])
        self.conv2      = nn.Conv2d(num_channels, num_classes, kernel_size=1)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)

        # Stubs obligatoires
        self.quant   = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x          = self.quant(x)                          # float32 → int8
        x          = self.sequential(x)
        activation = self.conv2(x)
        x          = self.avgpool(activation)
        out        = x.view(x.shape[0], -1)
        out        = self.dequant(out)                      # int8 → float32

        # Pour les evidence maps (inchangé)
        activation  = self.dequant(activation)
        bs, c, h, w = activation.shape
        att_weight  = torch.zeros((bs, c, h, w), device=activation.device)

        return out, activation, att_weight

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook


def _patched_bottleneck_forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)        # ← relu1 dédié (pas self.relu partagé)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)        # ← relu2 dédié

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    # ← remplacement du crop brutal par une gestion propre
    # ← remplacement du crop brutal par une gestion propre
    if residual.shape[2:] != out.shape[2:]:
        dh = (residual.size(2) - out.size(2)) // 2
        dw = (residual.size(3) - out.size(3)) // 2
        residual = residual[:, :, dh:dh+out.size(2), dw:dw+out.size(3)]

    out = self.skip_add.add(out, residual)
    out = self.relu3(out)        # ← relu3 dédié

    return out


def _patch_bottleneck_add(model):
    for module in model.modules():
        if isinstance(module, bpn.Bottleneck):
            module.skip_add = torch.ao.nn.quantized.FloatFunctional()

            # 3 nouveaux attributs ReLU ajoutés dynamiquement au bloc
            module.relu1 = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            module.relu3 = nn.ReLU(inplace=False)

def get_bagnet_model(num_classes:int) -> nn.Module:
    bpn.Bottleneck.forward = _patched_bottleneck_forward
    model = bagnet33(pretrained=False)
    fp32_bagnet_model = Bagnet(model=model, num_classes=3)
    _patch_bottleneck_add(fp32_bagnet_model)
    fp32_bagnet_model.conv2.register_forward_hook(get_activation('evidence_maps'))
    return fp32_bagnet_model