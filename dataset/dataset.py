import os
import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset
from typing import Tuple, Any, Callable, List, Optional, Union

class CustomDataset(Dataset):

    """
    커스텀 데이터셋 클래스.

    Args:
        root_dir (str): 이미지 파일들이 저장된 기본 디렉토리.
        info_df (pd.DataFrame): 이미지 경로 및 레이블 정보가 담긴 DataFrame.
        transform (Callable): 이미지에 적용될 변환 처리 함수.
        is_inference (bool, optional): 추론 모드 여부. 기본값은 False.
    """

    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target
        