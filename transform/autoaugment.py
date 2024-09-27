import torch
import numpy as np

from PIL import Image
from config import my_config
from torchvision import transforms

use_grayscale = True

class AutoaugmentTransform:

    """
    Torchvision의 transforms 모듈을 사용하여 AutoAugment 및 기타 변환을 적용하는 클래스. 

    Args:
        is_train (bool): 훈련 중인지 여부를 결정하는 플래그. 기본값은 True로, 훈련 모드에서는 추가적인 데이터 증강이 적용됨.
        size (tuple): 이미지를 리사이즈할 크기. 기본값은 (224, 224).
    """

    def __init__(self, is_train: bool = True, size: tuple=(224, 224)):

        """
        클래스 초기화 메서드. 훈련 모드(is_train)가 True일 경우 데이터 증강(AutoAugment)이 포함된 변환 파이프라인을 생성하고,
        훈련 모드가 아니면 일반적인 변환만 적용됩니다.

        Args:
            is_train (bool): 훈련 모드 여부를 결정하는 플래그. 기본값은 True.
            size (tuple): 리사이즈할 이미지 크기. 기본값은 (224, 224).
        """
        
        common_transforms = [
            transforms.Resize((my_config.image_size, my_config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ]
        
        if is_train:
            self.transform = transforms.Compose(
                [   
                    transforms.Grayscale(num_output_channels=3) if use_grayscale else transforms.Lambda(lambda x: x), 
                    transforms.AutoAugment(), 
                ] + common_transforms
            )
        else:
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:

        """
        이미지를 입력받아 변환을 적용하고, 변환된 이미지를 반환합니다.

        Args:
            image (np.ndarray): 변환을 적용할 이미지. NumPy 배열 형식이어야 합니다.

        Returns:
            torch.Tensor: 변환된 이미지 텐서.
        """
        
        image = Image.fromarray(image)
        transformed = self.transform(image)
        return transformed
