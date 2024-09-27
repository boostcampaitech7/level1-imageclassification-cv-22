import torch
import numpy as np

from PIL import Image
from config import my_config
from torchvision import transforms

class TorchvisionTransform:

    """
    Torchvision의 transforms 모듈을 사용하여 이미지에 다양한 변환을 적용하는 클래스.

    Args:
        is_train (bool): 훈련 중인지 여부를 결정하는 플래그. True일 경우 추가적인 데이터 증강이 적용됨.
        size (tuple): 이미지를 리사이즈할 크기. 기본값은 (224, 224).
    """

    def __init__(self, is_train: bool = True, size: tuple = (224, 224)):

        """
        클래스 초기화 메서드. 훈련 모드일 경우 설정된 변환 목록에 데이터 증강을 포함한 변환 파이프라인을 생성하고,
        훈련 모드가 아닐 경우 기본적인 변환만 적용됩니다.

        Args:
            is_train (bool): 훈련 모드 여부를 결정하는 플래그. 기본값은 True.
            size (tuple): 이미지 크기를 조정할 크기. 기본값은 (224, 224).
        """

        common_transforms = [
            transforms.Resize((my_config.image_size, my_config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if is_train:
            self.transform = transforms.Compose(
                my_config.train_transforms + common_transforms
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
