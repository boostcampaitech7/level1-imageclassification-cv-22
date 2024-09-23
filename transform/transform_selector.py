from .torchvision_transform import TorchvisionTransform
from .albumentations_transform import AlbumentationsTransform

class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """
    def __init__(self, transform_type: str):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations"]:
            self.transform_type = transform_type
        
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool, size: tuple):
        
        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.transform_type == 'torchvision':
            transform = TorchvisionTransform(is_train=is_train, size=size)
        
        elif self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train, size=size)
        
        return transform