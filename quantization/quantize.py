from quantization.ptq_resnet import ptq_quantization_resnet


def quantize(cfg, model, calib_loader, n_batches):
    bagnet = True if 'bagnet' in cfg["training"]["network"] else False
    if bagnet:
        return "Not implemented"
    else:
        resnet_int8_model =  ptq_quantization_resnet(model, calib_loader, n_batches)
        return resnet_int8_model