import torchvision.models as models
import torch.nn as nn

def get_resnet_model(num_classes: int = 3) -> nn.Module:
    """Return a ResNet-50 with a custom classification head."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model