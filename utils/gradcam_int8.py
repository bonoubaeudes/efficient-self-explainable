import torch


class GradCAM_INT8:
    """
    Forward-only CAM for static INT8 quantized ResNets.
    Uses the linear classifier weights as a proxy for gradients (CAM-style).
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model   = model.eval().to(device)
        self.device  = device
        self._features: dict = {}

        # Find the last layer4 block in the FX graph
        hook_target = None
        for name, module in self.model.named_modules():
            if "layer4" in name and "relu" not in name.lower():
                hook_target = (name, module)

        if hook_target is None:
            raise RuntimeError("layer4 not found in the quantized model.")

        print(f"  Hook attached on: {hook_target[0]}  ({type(hook_target[1]).__name__})")
        self._hook = hook_target[1].register_forward_hook(
            lambda m, inp, out: self._features.update({"feat": out})
        )

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None):
        self._features.clear()
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        if "feat" not in self._features:
            raise RuntimeError("Hook did not capture features.")

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        feat = self._features["feat"]
        if feat.is_quantized:
            feat = feat.dequantize()
        f = feat[0].float()   # [C, H, W]

        try:
            w_c = self.model.fc.weight().dequantize()[class_idx].float()
        except TypeError:
            w_c = self.model.fc.weight[class_idx].float()

        cam = torch.relu(torch.einsum("c,chw->hw", w_c, f))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx

    def remove_hook(self):
        self._hook.remove()