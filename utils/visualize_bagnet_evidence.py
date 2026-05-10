import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
import numpy as np
import os

from data.dataset import get_transforms


def visualize_bagnet_evidence(cfg, image_name, model, class_idx=None, alpha=0.5):
    _, test_transform = get_transforms(
        image_size  = cfg["data"]["image_size"],
        crop_size   = cfg["data"]["crop_size"],
    )
    data_dir = cfg["data"]["data_dir"]
    image_path = os.path.join(data_dir, image_name)
    img = Image.open(image_path).convert("RGB")
    img = img.resize(
        (cfg["data"]["image_size"], cfg["data"]["image_size"])
    )
    image_tensor = test_transform(img).unsqueeze(0).to("cpu")

    model.eval()
    with torch.no_grad():
        out, activation, _ = model(image_tensor)

    if class_idx is None:
        class_idx = out.argmax(dim=1).item()

    evidence = activation[0, class_idx].cpu()  # [h, w]

    evidence = evidence - evidence.min()
    evidence = evidence / (evidence.max() + 1e-8)

    H, W = img.size[1], img.size[0]
    evidence_up = F.interpolate(
        evidence.unsqueeze(0).unsqueeze(0),  # [1, 1, h, w]
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).squeeze().numpy()  # [H, W]

    # Colormap
    heatmap = cm.jet(evidence_up)[:, :, :3]  # [H, W, 3], values [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)


    img_np = np.array(img)
    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    #overlay = cv2.addWeighted(
    #    heatmap,
    #    alpha,
    #    img_np,
    #    1 - alpha,
    #    0
    #)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Image ")
    axes[0].axis("off")

    axes[1].imshow(evidence_up, cmap='jet')
    axes[1].set_title(f"Evidence map (class {class_idx})")
    axes[1].axis("off")
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("evidence_map.png", dpi=150, bbox_inches='tight')
    plt.show()