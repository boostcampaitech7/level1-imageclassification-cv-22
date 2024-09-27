import timm
import torch
import torch.nn as nn

class TimmModel(nn.Module):

    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스. 

    Args:
        model_name (str): 사용할 timm 모델의 이름.
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
        TimmModel 클래스의 생성자.

        Timm 라이브러리를 사용해 지정된 사전 훈련된 모델을 초기화합니다.

        Args:
            model_name (str): 사용할 timm 모델의 이름.
            num_classes (int): 분류할 클래스의 개수.
            pretrained (bool): 사전 훈련된 가중치를 사용할지 여부.
        """
    
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        """
        순전파 함수. 입력 데이터를 timm 모델을 통해 전달하여 최종 출력값을 생성합니다.

        Args:
            x (torch.Tensor): 모델에 입력할 이미지 텐서. (배치 크기, 채널, 높이, 너비).

        Returns:
            torch.Tensor: 모델의 최종 출력값.
        """
    
        return self.model(x)