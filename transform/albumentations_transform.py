import torch
import numpy as np
import albumentations as A

from config import my_config
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:

    """
    Albumentations 라이브러리를 사용하여 이미지에 다양한 변환을 적용하는 클래스. 

    Args:
        is_train (bool): 훈련 중인지 여부를 결정하는 플래그. 기본값은 True로, 훈련 모드에서는 추가적인 데이터 증강이 적용됨.
    """

    def __init__(self, is_train: bool = True):

        """
        클래스 초기화 메서드. 훈련 모드(is_train)가 True일 경우 데이터 증강을 위한 추가 변환들이 포함된 변환 파이프라인을 생성하고,
        훈련 모드가 아니면 일반적인 변환만 적용됩니다.

        Args:
            is_train (bool): 훈련 모드 여부를 결정하는 플래그. 기본값은 True.
        """

        common_transforms = [
            A.Resize(my_config.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]
        
        if is_train:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15),
                    A.RandomBrightnessContrast(p=0.2),
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
 
        """
        이미지를 입력받아 변환을 적용하고, 변환된 이미지를 반환합니다.

        Args:
            image (np.ndarray): 변환을 적용할 이미지. NumPy 배열 형식이어야 합니다.

        Returns:
            torch.Tensor: 변환된 이미지 텐서.

        Raises:
            TypeError: 입력 이미지가 NumPy 배열이 아닐 경우 발생.
        """

        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        transformed = self.transform(image=image)
        
        return transformed['image']
