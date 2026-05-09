from models.resnet import get_model


def builder(model, num_classes:int):
    if model == "resnet":
        return get_model(num_classes = num_classes)
    elif model == "bagnet":
        return "to implement"
