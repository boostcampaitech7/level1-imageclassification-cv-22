import timm
import torch
import torch.nn as nn

class TimmModel(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool
    ):
        super(TimmModel, self).__init__()
        print("모델 : ",  model_name)
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)