import torch.nn.functional as F
import torch



# ---------------------------------------------------------------------------
# GradCAM for FP32 / FP16 ResNet
# ---------------------------------------------------------------------------

class GradCAM_FP32:
    """
    Gradient-based CAM. Hooks onto model.layer4.
    Compatible with FP32 and FP16 ResNet-50.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model   = model.eval().to(device)
        self.device  = device
        self._features:  dict = {}
        self._gradients: dict = {}

        self.model.layer4.register_forward_hook(
            lambda m, inp, out: self._features.update({"feat": out})
        )
        self.model.layer4.register_full_backward_hook(
            lambda m, gin, gout: self._gradients.update({"grad": gout[0]})
        )

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Returns:
            cam (np.ndarray):  normalized heatmap, shape (H, W)
            class_idx (int):   predicted class index
        """
        self._features.clear()
        self._gradients.clear()
        input_tensor = input_tensor.to(self.device)

        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        output[0, class_idx].backward()

        feat = self._features["feat"][0].float()   # [C, H, W]
        grad = self._gradients["grad"][0].float()  # [C, H, W]

        weights = grad.mean(dim=(1, 2))            # GAP on gradients → [C]
        cam     = torch.relu(torch.einsum("c,chw->hw", weights, feat))

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().detach().numpy(), class_idx

    def remove_hook(self):
        pass  # hooks are auto-managed