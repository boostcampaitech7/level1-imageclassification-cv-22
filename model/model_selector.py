from .simple_cnn import SimpleCNN
from .timm_model import TimmModel
from .torchvision_model import TorchvisionModel

import torch.nn as nn

class ModelSelector:
    """
    사용할 모델 유형을 선택하는 클래스.
    """
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):
        
        print("모델 : ", model_type)
        if model_type == 'simple':
            self.model = SimpleCNN(num_classes=num_classes)
        
        elif model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        # 생성된 모델 객체 반환
        return self.model