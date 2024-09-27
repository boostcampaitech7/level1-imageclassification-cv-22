from .autoaugment import AutoaugmentTransform
from .torchvision_transform import TorchvisionTransform
from .albumentations_transform import AlbumentationsTransform

class TransformSelector:

    """
    주어진 변환 유형에 따라 적절한 데이터 변환 클래스를 선택하는 클래스. 

    Args:
        transform_type (str): 사용할 변환 라이브러리의 이름. 'torchvision', 'albumentations', 'autoaugment' 중 하나를 선택해야 함.

    Raises:
        ValueError: 지정된 변환 라이브러리가 알 수 없는 경우 발생.
    """

    def __init__(self, transform_type: str):
    
        """
        클래스 초기화 메서드. 제공된 변환 라이브러리가 유효한지 확인하고, 그렇지 않으면 오류를 발생시킵니다.

        Args:
            transform_type (str): 사용할 변환 라이브러리의 이름 ('torchvision', 'albumentations', 'autoaugment') 중 하나여야 합니다.

        Raises:
            ValueError: 잘못된 변환 라이브러리가 지정된 경우 발생.
        """
    
        if transform_type in ["torchvision", "albumentations", "autoaugment"]:
            self.transform_type = transform_type
        else:
            raise ValueError("Unknown transformation library specified.")

    def get_transform(self, is_train: bool):
    
        """
        설정된 변환 라이브러리에 따라 적절한 변환 클래스를 반환합니다.

        Args:
            is_train (bool): 훈련 모드 여부를 나타내는 플래그. True일 경우 훈련용 변환을 적용.

        Returns:
            transform: 선택된 변환 라이브러리의 인스턴스를 반환합니다.
        """
    
        if self.transform_type == 'torchvision':
            transform = TorchvisionTransform(is_train=is_train)
        elif self.transform_type == 'albumentations':
            transform = AlbumentationsTransform(is_train=is_train)
        elif self.transform_type == 'autoaugment':
            transform = AutoaugmentTransform(is_train=is_train)
        return transform
