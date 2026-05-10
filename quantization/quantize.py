from quantization.ptq_bagnet import ptq_quantize_static_bagnet
from quantization.ptq_resnet import ptq_quantization_resnet
from utils.func import save_weights


def quantize(cfg, model, calib_loader, n_batches):
    bagnet = True if 'bagnet' in cfg["training"]["network"] else False
    if bagnet:
        path = cfg["paths"]["bagnet_int8_checkpoint"]
        bagnet_int8_model = ptq_quantize_static_bagnet(model, calib_loader, n_batches)
        save_weights(bagnet_int8_model, path)
        return bagnet_int8_model
    else:
        path = cfg["paths"]["resnet_int8_checkpoint"]
        resnet_int8_model =  ptq_quantization_resnet(model, calib_loader, n_batches)
        save_weights(resnet_int8_model, path)
        return resnet_int8_model