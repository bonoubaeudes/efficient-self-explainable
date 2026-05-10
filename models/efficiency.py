import copy

from models.load_models import from_fp32_to_fp16, load_fp32bagnet_model, load_int8bagnet_model, load_fp32resnet_model, \
    load_int8resnet_model
from utils.metrics import evaluate_accuracy, model_mb, measure_inference_time


def compute_all_model_efficiency(cfg, num_classes:int, test_loader):
    fp32bagnet_model = load_fp32bagnet_model(cfg, num_classes)
    int8bagnet_model = load_int8bagnet_model(cfg, num_classes)
    result_bagnet = efficiency(fp32bagnet_model, int8bagnet_model, test_loader, True)

    fp32resnet_model = load_fp32resnet_model(cfg, num_classes)
    int8resnet_model = load_int8resnet_model(cfg, num_classes)
    results_resnet = efficiency(fp32resnet_model, int8resnet_model, test_loader, False)
    all_results = {
        'Dense BagNet':  result_bagnet,
        'Sparse BagNet': result_bagnet,
        'ResNet':        results_resnet,
    }
    return all_results


def efficiency(fp32model, int8model, test_loader, is_bagnet=True):
    fp16model = from_fp32_to_fp16(fp32model)
    variants = {
        'FP32':     (fp32model,  'cuda', 'fp32'),
        'FP16':     (fp16model,  'cuda', 'fp16'),
        'INT8':     (int8model,  'cpu',  'int8'),
    }
    results = {}
    for name, (mdl, dev, prec) in variants.items():
        print(f"\n── {name} ──")

        acc  = evaluate_accuracy(mdl, test_loader, device=dev, precision=prec, is_bagnet=is_bagnet)
        size = model_mb(mdl)

        # ── PC inference ──
        t_pc_mean, t_pc_std = measure_inference_time(
            mdl, device=dev, precision=prec,
            input_size=(1, 3, 224, 224))

        # ── Edge inference (CPU uniquement) ──
        if prec == 'int8':
            # INT8 reste tel quel sur CPU
            edge_mdl  = mdl
            edge_prec = 'int8'
        elif prec == 'fp16':
            # FP16 → convertit en FP32 pour CPU
            edge_mdl  = copy.deepcopy(mdl).float().cpu()
            edge_prec = 'fp32'
        else:
            # FP32 → déplace sur CPU
            edge_mdl  = copy.deepcopy(mdl).float().cpu()
            edge_prec = 'fp32'

        t_edge_mean, t_edge_std = measure_inference_time(
            edge_mdl, device='cpu', precision=edge_prec,
            input_size=(1, 3, 224, 224))

        results[name] = {
            'acc':       acc,
            'size':      size,
            'pc_mean':   t_pc_mean  or 0.0,
            'pc_std':    t_pc_std   or 0.0,
            'edge_mean': t_edge_mean or 0.0,
            'edge_std':  t_edge_std  or 0.0,
        }
        print(f"  Accuracy    : {acc:.2f}%")
        print(f"  Size        : {size} MB")
        print(f"  PC inference: {t_pc_mean:.1f} ± {t_pc_std:.1f} ms")
        print(f"  Edge infer  : {t_edge_mean:.1f} ± {t_edge_std:.1f} ms")

        # Libère la mémoire de la copie edge
        if prec in ('fp16', 'fp32'):
            del edge_mdl
        return results