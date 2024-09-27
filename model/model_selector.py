import torch.nn as nn

from .simple_cnn import SimpleCNN
from .timm_model import TimmModel
from .torchvision_model import TorchvisionModel

class ModelSelector:

    """
    사용할 모델 유형을 선택하는 클래스.

    Args:
        model_type (str): 사용할 모델의 유형 ('simple', 'torchvision', 'timm').
        num_classes (int): 분류할 클래스의 개수.
        **kwargs: 특정 모델에 추가적으로 전달할 파라미터.
    """
    
    def __init__(
        self, 
        model_type: str, 
        num_classes: int, 
        **kwargs
    ):

        """
        ModelSelector 클래스의 생성자.

        모델 유형에 따라 적절한 모델 객체를 생성합니다.

        Args:
            model_type (str): 사용할 모델의 유형.
            num_classes (int): 분류할 클래스의 개수.
            **kwargs: 추가 파라미터.
        
        Raises:
            ValueError: 지원되지 않는 모델 유형이 지정된 경우 예외 발생.
        """

        if model_type == 'simple':
            self.model = SimpleCNN(num_classes=num_classes)
        
        elif model_type == 'torchvision':
            self.model = TorchvisionModel(num_classes=num_classes, **kwargs)
        
        elif model_type == 'timm':
            self.model = TimmModel(num_classes=num_classes, **kwargs)
        
        else:
            raise ValueError("Unknown model type specified.")

    def get_model(self) -> nn.Module:

        """
        생성된 모델 객체를 반환합니다. 

        Returns:
            nn.Module: 선택된 모델 객체.
        """
        
        return self.model