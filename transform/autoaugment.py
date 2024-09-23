import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from config import my_config

use_grayscale = True

class AutoaugmentTransform:
    def __init__(self, is_train: bool = True, size: tuple=(224, 224)):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 정규화
        common_transforms = [
            transforms.Resize((my_config.image_size, my_config.image_size)),  # 이미지를 224x224 크기로 리사이즈
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
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
        image = Image.fromarray(image)
        transformed = self.transform(image)
        return transformed  
