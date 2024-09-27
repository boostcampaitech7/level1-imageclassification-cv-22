import torch
import torch.nn as nn

from torchvision import models

class TorchvisionModel(nn.Module):
    
    """
    Torchvision에서 제공하는 사전 훈련된 모델을 사용하는 클래스. 

    Args:
        model_name (str): 사용할 torchvision 모델의 이름.
        num_classes (int): 분류할 클래스의 개수.
        pretrained (bool): 사전 훈련된 가중치를 사용할지 여부.
    """
    
    def __init__(
        self, 
        model_name: str, 
        num_classes: int, 
        pretrained: bool
    ):
    
        """
        TorchvisionModel 클래스의 생성자.

        지정된 torchvision 모델을 로드하고, 마지막 분류기를 사용자 정의 클래스 수에 맞게 조정합니다.

        Args:
            model_name (str): 사용할 torchvision 모델의 이름.
            num_classes (int): 분류할 클래스의 개수.
            pretrained (bool): 사전 훈련된 가중치를 사용할지 여부.
        """
    
        super(TorchvisionModel, self).__init__()
        self.model = models.__dict__[model_name](pretrained=pretrained)
        self.model_name = model_name
        
        if 'fc' in dir(self.model):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        elif 'classifier' in dir(self.model):
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        """
        순전파 함수. 입력 데이터를 torchvision 모델을 통해 전달하여 최종 출력값을 생성합니다.

        Args:
            x (torch.Tensor): 모델에 입력할 이미지 텐서. (배치 크기, 채널, 높이, 너비).

        Returns:
            torch.Tensor: 모델의 최종 출력값.
        """
    
        return self.model(x)