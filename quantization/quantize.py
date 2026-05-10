from quantization.ptq_bagnet import ptq_quantize_static_bagnet
from quantization.ptq_resnet import ptq_quantization_resnet


def quantize(cfg, model, calib_loader, n_batches):
    bagnet = True if 'bagnet' in cfg["training"]["network"] else False
    if bagnet:
        bagnet_int8_model = ptq_quantize_static_bagnet(model, calib_loader, n_batches)
        return bagnet_int8_model
    else:
        resnet_int8_model =  ptq_quantization_resnet(model, calib_loader, n_batches)
        return resnet_int8_model