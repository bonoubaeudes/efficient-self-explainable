import copy

from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from models.load_models import load_fp32bagnet_model, load_int8bagnet_model, load_fp32resnet_model, from_fp32_to_fp16, \
    load_int8resnet_model
from utils.gradcam_fp32_fp16 import GradCAM_FP32
from utils.gradcam_int8 import GradCAM_INT8



def extract_topk_regions_bagnet(evidence_map, image_tensor, k=5, region_size=38):
    """
    evidence_map : torch.Tensor [H, W]  — upsampled saliency map (0–1)
    image_tensor : torch.Tensor [C, H, W]
    k            : number of top regions to extract
    region_size  : size of each square patch (pixels)
    """
    H, W   = evidence_map.shape
    half_lo = region_size // 2
    half_hi = region_size - half_lo
    stride  = max(region_size // 8, 4)   # stride adaptatif plus fin

    scores, coords = [], []

    for y in range(half_lo, H - half_hi + 1, stride):
        for x in range(half_lo, W - half_hi + 1, stride):
            region_sal = evidence_map[y - half_lo : y + half_hi,
                         x - half_lo : x + half_hi]
            # Vérifie que le patch est complet
            if region_sal.shape != (region_size, region_size):
                continue
            scores.append(region_sal.mean().item())
            coords.append((x, y))

    if len(scores) == 0:
        return []

    order    = np.argsort(scores)[::-1]
    selected = []
    used_centers = []

    for idx in order:
        cx, cy = coords[idx]
        # NMS : distance euclidienne entre centres
        too_close = any(
            ((cx - px) ** 2 + (cy - py) ** 2) < (region_size ** 2)
            for px, py in used_centers
        )
        if not too_close:
            selected.append(idx)
            used_centers.append((cx, cy))
        if len(selected) == k:
            break

    regions = []
    for rank, idx in enumerate(selected):
        cx, cy = coords[idx]
        x1 = max(cx - half_lo, 0)
        y1 = max(cy - half_lo, 0)
        x2 = min(cx + half_hi, W)
        y2 = min(cy + half_hi, H)
        patch = image_tensor[:, y1:y2, x1:x2]
        regions.append({
            'rank'  : rank + 1,
            'score' : scores[idx],
            'bbox'  : (x1, y1, x2 - x1, y2 - y1),
            'patch' : patch,
            'center': (cx, cy),
        })
    return regions


def compute_dataset_sensitivity_curve_bagnet(dataset, model, transform,
                                      k_max=10, region_size=None,
                                      n_images=None, shuffle=True,
                                      seed=42, device='cuda',precision='fp32'):

    # ── Config précision ──
    precision = precision.lower()
    use_fp16  = precision == 'fp16'
    use_int8  = precision == 'int8'

    if use_int8:
        device = 'cpu'    # PTQ statique → CPU obligatoire
        print("  [INT8] forcé sur CPU")

    model.eval()
    model.to(device)

    if use_fp16:
        model = model.half()
        print("  [FP16] modèle casté en float16")

    all_curves_abs = []
    all_curves_rel = []
    skipped = 0

    # ── region_size auto sur la 1ère image ──
    _img0, _ = dataset[0]
    if isinstance(_img0, torch.Tensor):
        _t0 = _img0.unsqueeze(0)
    else:
        _t0 = transform(_img0).unsqueeze(0)
    if use_fp16:
        _t0 = _t0.half()
    _t0 = _t0.to(device)
    _H, _W = _t0.shape[-2], _t0.shape[-1]
    with torch.no_grad():
        _, _act0, _ = model(_t0)
    _feat_px = _H / _act0.shape[2]
    if region_size is None:
        region_size = max(32, int(_feat_px * 4))
    print(f"  precision   : {precision.upper()}")
    print(f"  region_size : {region_size}px | image : {_H}x{_W} | "
          f"activation : {_act0.shape[2]}x{_act0.shape[3]}")
    del _t0, _act0

    for item in tqdm(dataset, desc=f"Sensitivity [{precision.upper()}]"):
        img = item[0].convert("RGB")

        image_tensor = transform(img).unsqueeze(0)
        if use_fp16:
            image_tensor = image_tensor.half()
        image_tensor = image_tensor.to(device)

        H, W = image_tensor.shape[-2], image_tensor.shape[-1]

        with torch.no_grad():
            out, activation, _ = model(image_tensor)

        if activation is None or activation.ndim < 4:
            skipped += 1; continue

        probs      = F.softmax(out.float(), dim=1)[0]   # .float() → stable pour fp16
        pred_class = probs.argmax().item()
        score_orig = probs[pred_class].item()

        # ── Saliency map sur pred_class ──
        evidence = activation[0, pred_class].cpu().float()   # toujours float32
        evidence = torch.relu(evidence)
        if evidence.max() < 1e-8:
            skipped += 1; continue
        evidence = evidence / (evidence.max() + 1e-8)

        evidence_up = F.interpolate(
            evidence.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze()

        img_np  = np.array(img.resize((W, H)))
        regions = extract_topk_regions_bagnet(
            evidence_up,
            image_tensor.squeeze(0).cpu().float(),   # .float() pour fp16
            k=k_max, region_size=region_size
        )
        if len(regions) < k_max:
            skipped += 1; continue

        curve_abs = [score_orig]
        curve_rel = [0.0]
        masked_np = img_np.copy()

        for k in range(1, k_max + 1):
            x, y, w, h = regions[k - 1]['bbox']
            masked_np[y:y+h, x:x+w] = 0
            t = transform(Image.fromarray(masked_np)).unsqueeze(0)
            if use_fp16:
                t = t.half()
            t = t.to(device)
            with torch.no_grad():
                out_m, _, _ = model(t)
            score_k = F.softmax(out_m.float(), dim=1)[0, pred_class].item()
            curve_abs.append(score_k)
            curve_rel.append((score_orig - score_k) / (score_orig + 1e-8))

        all_curves_abs.append(curve_abs)
        all_curves_rel.append(curve_rel)

    print(f"\n  Traitées : {len(all_curves_abs)} / {len(dataset)}  (skippées : {skipped})")
    if len(all_curves_abs) == 0:
        raise RuntimeError("Aucune image traitée.")

    abs_arr = np.array(all_curves_abs)
    rel_arr = np.array(all_curves_rel)

    return {
        'k':          list(range(abs_arr.shape[1])),
        'mean_score': abs_arr.mean(axis=0).tolist(),
        'std_score':  abs_arr.std(axis=0).tolist(),
        'mean_drop':  rel_arr.mean(axis=0).tolist(),
        'std_drop':   rel_arr.std(axis=0).tolist(),
        'per_image':  abs_arr,
        'n_images':   len(abs_arr),
        'precision':  precision.upper(),
    }

def extract_topk_regions_resnet(evidence_map, k=5, region_size=38):
    """
    evidence_map : torch.Tensor [H, W]  — upsampled saliency map (0–1)
    k            : number of top regions to extract
    region_size  : size of each square patch (pixels)
    """
    H, W    = evidence_map.shape
    half_lo = region_size // 2
    half_hi = region_size - half_lo
    stride  = max(region_size // 8, 4)

    scores, coords = [], []
    for y in range(half_lo, H - half_hi + 1, stride):
        for x in range(half_lo, W - half_hi + 1, stride):
            region_sal = evidence_map[y - half_lo : y + half_hi,
                         x - half_lo : x + half_hi]
            if region_sal.shape != (region_size, region_size):
                continue
            scores.append(region_sal.mean().item())
            coords.append((x, y))

    if len(scores) == 0:
        return []

    order    = np.argsort(scores)[::-1]
    selected = []
    used_centers = []
    for idx in order:
        cx, cy = coords[idx]
        too_close = any(
            ((cx - px) ** 2 + (cy - py) ** 2) < (region_size ** 2)
            for px, py in used_centers
        )
        if not too_close:
            selected.append(idx)
            used_centers.append((cx, cy))
        if len(selected) == k:
            break

    regions = []
    for rank, idx in enumerate(selected):
        cx, cy = coords[idx]
        x1 = max(cx - half_lo, 0)
        y1 = max(cy - half_lo, 0)
        x2 = min(cx + half_hi, W)
        y2 = min(cy + half_hi, H)
        regions.append({
            'rank'  : rank + 1,
            'score' : scores[idx],
            'bbox'  : (x1, y1, x2 - x1, y2 - y1),
            'center': (cx, cy),
        })
    return regions

def compute_dataset_sensitivity_curve_resnet(dataset, model, transform,
                                             k_max=10, region_size=None,
                                             n_images=None, seed=42,
                                             device='cuda', precision='fp32'):
    precision = precision.lower()
    use_fp16  = precision == 'fp16'
    use_int8  = precision == 'int8'

    if use_int8:
        device = 'cpu'
        print("  [INT8] forcé sur CPU")

    model.eval().to(device)
    if use_fp16:
        model = model.half()
        print("  [FP16] modèle casté en float16")

    # ── Instanciation GradCAM selon précision ──
    if use_int8:
        gradcam = GradCAM_INT8(model, device=device)
    else:
        gradcam = GradCAM_FP32(model, device=device)

    # ── region_size auto sur 1ère image ──
    _img0, _ = dataset[0]
    if isinstance(_img0, torch.Tensor):
        _t0 = _img0.unsqueeze(0)
    else:
        _t0 = transform(_img0).unsqueeze(0)
    if use_fp16:
        _t0 = _t0.half()
    _t0 = _t0.to(device)
    _H, _W = _t0.shape[-2], _t0.shape[-1]
    if region_size is None:
        region_size = max(38, _H // 7)
    print(f"  precision   : {precision.upper()}")
    print(f"  region_size : {region_size}px | image : {_H}x{_W}")
    del _t0

    all_curves_abs = []
    all_curves_rel = []

    # ── Compteurs de skip par cause ──
    skip_forward   = 0
    skip_gradcam   = 0
    skip_cam_zero  = 0
    skip_regions   = 0

    for img, label in tqdm(dataset, desc=f"Sensitivity ResNet [{precision.upper()}]"):
        if isinstance(img, torch.Tensor):
            image_tensor = img.unsqueeze(0)
        else:
            image_tensor = transform(img).unsqueeze(0)

        if use_fp16:
            image_tensor = image_tensor.half()
        image_tensor = image_tensor.to(device)

        H, W = image_tensor.shape[-2], image_tensor.shape[-1]

        # ── Première passe : pred_class + score_orig ──
        try:
            with torch.no_grad():
                out0 = model(image_tensor)
            logits0 = out0[0] if isinstance(out0, (tuple, list)) else out0
            if hasattr(logits0, 'dequantize'):
                logits0 = logits0.dequantize()
            probs      = F.softmax(logits0.float(), dim=1)[0]
            pred_class = probs.argmax().item()
            score_orig = probs[pred_class].item()
        except Exception as e:
            if skip_forward == 0:
                print(f"\n  [SKIP forward] {type(e).__name__}: {e}")
            skip_forward += 1
            continue

        # ── GradCAM via la classe ──
        try:
            cam_np, _ = gradcam.generate_cam(image_tensor, class_idx=pred_class)
            cam = torch.from_numpy(cam_np).float()
        except Exception as e:
            if skip_gradcam == 0:
                print(f"\n  [SKIP gradcam] {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
            skip_gradcam += 1
            continue

        if cam.max() < 1e-8:
            if skip_cam_zero == 0:
                print(f"\n  [SKIP cam_zero] cam.max()={cam.max():.2e}  ")
            skip_cam_zero += 1
            continue

        cam_up = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze()

        img_np  = np.array(img.resize((W, H)))
        regions = extract_topk_regions_resnet(
            cam_up,
            k=k_max, region_size=region_size
        )
        if len(regions) < k_max:
            if skip_regions == 0:
                print(f"\n  [SKIP regions] only {len(regions)}/{k_max} regions found ")
                print(f"    cam_up shape={cam_up.shape}  region_size={region_size}  H={H} W={W}")
            skip_regions += 1
            continue

        curve_abs = [score_orig]
        curve_rel = [0.0]
        masked_np = img_np.copy()

        for k in range(1, k_max + 1):
            x, y, w, h = regions[k - 1]['bbox']
            masked_np[y:y+h, x:x+w] = 0
            t = transform(Image.fromarray(masked_np)).unsqueeze(0)
            if use_fp16:
                t = t.half()
            t = t.to(device)
            with torch.no_grad():
                out_m = model(t)
            logits_m = out_m[0] if isinstance(out_m, (tuple, list)) else out_m
            if hasattr(logits_m, 'dequantize'):
                logits_m = logits_m.dequantize()
            score_k = F.softmax(logits_m.float(), dim=1)[0, pred_class].item()
            curve_abs.append(score_k)
            curve_rel.append((score_orig - score_k) / (score_orig + 1e-8))

        all_curves_abs.append(curve_abs)
        all_curves_rel.append(curve_rel)

    # ── Cleanup hooks ──
    gradcam.remove_hook()

    # ── Rapport de skip ──
    total   = len(dataset)
    treated = len(all_curves_abs)
    skipped = skip_forward + skip_gradcam + skip_cam_zero + skip_regions
    print(f"\n  Traitées : {treated} / {total}  (skippées : {skipped})")
    print(f"    ├─ forward error  : {skip_forward}")
    print(f"    ├─ gradcam error  : {skip_gradcam}")
    print(f"    ├─ cam all-zero   : {skip_cam_zero}")
    print(f"    └─ regions < k    : {skip_regions}")

    if treated == 0:
        raise RuntimeError(
            f"Aucune image traitée sur {total}. "
            f"Causes : forward={skip_forward}, gradcam={skip_gradcam}, "
            f"cam_zero={skip_cam_zero}, regions={skip_regions}."
        )

    abs_arr = np.array(all_curves_abs)
    rel_arr = np.array(all_curves_rel)

    return {
        'k':          list(range(abs_arr.shape[1])),
        'mean_score': abs_arr.mean(axis=0).tolist(),
        'std_score':  abs_arr.std(axis=0).tolist(),
        'mean_drop':  rel_arr.mean(axis=0).tolist(),
        'std_drop':   rel_arr.std(axis=0).tolist(),
        'per_image':  abs_arr,
        'n_images':   len(abs_arr),
        'precision':  precision.upper(),
    }

def compute_dataset_sensitivity_curve(cfg, num_classes, dataset, transform):
    fp32bagnet_model = load_fp32bagnet_model(cfg, num_classes)
    int8bagnet_model = load_int8bagnet_model(cfg, num_classes)
    results_fp32_dense_bagnet = compute_dataset_sensitivity_curve_bagnet(
        dataset, fp32bagnet_model, transform,
        k_max=10, n_images=1000, seed=42,
        device='cuda', precision='fp32')

    results_fp16_dense_bagnet = compute_dataset_sensitivity_curve_bagnet(
        dataset, copy.deepcopy(fp32bagnet_model), transform,
        k_max=10, n_images=1000, seed=42,
        device='cuda', precision='fp16')

    results_int8_dense_bagnet = compute_dataset_sensitivity_curve_bagnet(
        dataset, int8bagnet_model, transform,
        k_max=10, n_images=1000, seed=42,
        device='cpu', precision='int8')

    results_dense_bagnet = {
        'FP32': results_fp32_dense_bagnet,
        'FP16': results_fp16_dense_bagnet,
        'INT8': results_int8_dense_bagnet
    }
    fp32_resnet = load_fp32resnet_model(cfg, num_classes)
    fp16_resnet = from_fp32_to_fp16(fp32_resnet)
    int8_resnet = load_int8resnet_model(cfg, num_classes)
    results_resnet = {
        'FP32': compute_dataset_sensitivity_curve_resnet(
            dataset, fp32_resnet, transform,
            n_images=5, device='cuda', precision='fp32'),
        'FP16': compute_dataset_sensitivity_curve_resnet(
            dataset, fp16_resnet, transform,
            n_images=5, device='cuda', precision='fp16'),
        'INT8': compute_dataset_sensitivity_curve_resnet(
            dataset, int8_resnet, transform,
            n_images=5, device='cpu',  precision='int8'),
    }
    return results_dense_bagnet, results_resnet