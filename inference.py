import os
import torch
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.dataset import CustomDataset
from model.model_selector import ModelSelector
from transform.transform_selector import TransformSelector

class ModelInference:
    def __init__(self, 
                 testdata_dir, 
                 testdata_info_file, 
                 save_result_path, 
                 model_name='resnet18', 
                 batch_size=64,
                 num_classes=500, 
                 pretrained=False,
                 num_workers=6):
        
        """
        ModelInference 클래스 초기화 메서드. 모델, 데이터 경로, 배치 크기 등의 설정을 정의합니다. 

        Args:
            testdata_dir (str): 테스트 데이터 디렉토리 경로.
            testdata_info_file (str): 테스트 데이터 정보 파일 경로.
            save_result_path (str): 결과를 저장할 경로.
            model_name (str): 사용할 모델 이름.
            batch_size (int): 데이터 로딩 시 사용할 배치 크기.
            num_classes (int): 출력 클래스 수.
            pretrained (bool): 사전 학습된 모델을 사용할지 여부.
            num_workers (int): 데이터 로딩에 사용할 워커 수.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.testdata_dir = testdata_dir
        self.testdata_info_file = testdata_info_file
        self.save_result_path = save_result_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_workers = num_workers
        self.model = None
        self.test_loader = None

    def prepare_data(self):
        
        """
        테스트 데이터를 로드하고, 데이터셋과 데이터 로더를 준비하는 메서드.

        Returns:
            pd.DataFrame: 테스트 데이터 정보 파일을 판다스 데이터프레임으로 반환.
        """
        
        test_info = pd.read_csv(self.testdata_info_file)

        transform_selector = TransformSelector(transform_type="torchvision")
        test_transform = transform_selector.get_transform(is_train=False)

        test_dataset = CustomDataset(
            root_dir=self.testdata_dir,
            info_df=test_info,
            transform=test_transform,
            is_inference=True
        )

        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=self.num_workers
        )
        return test_info

    def prepare_model(self):
        
        """
        모델을 선택하고, 사전 학습된 가중치를 불러와 모델을 준비하는 메서드.
        """
        
        model_selector = ModelSelector(
            model_type='timm', 
            num_classes=self.num_classes,
            model_name=self.model_name, 
            pretrained=self.pretrained
        )
        self.model = model_selector.get_model()

        model_path = os.path.join(self.save_result_path, self.model.model_name,"best_model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.model.to(self.device)
        self.model.eval()

    def inference(self):
        
        """
        모델 추론을 수행하는 메서드.

        Returns:
            list: 예측된 클래스 레이블 목록.
        """
        
        predictions = []
        with torch.no_grad():
            for images in tqdm(self.test_loader):
                images = images.to(self.device)
                logits = self.model(images)
                logits = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                predictions.extend(preds.cpu().detach().numpy())

        return predictions

    def save_predictions(self, predictions, test_info):
        
        """
        추론 결과를 CSV 파일로 저장하는 메서드.

        Args:
            predictions (list): 예측된 클래스 레이블 목록.
            test_info (pd.DataFrame): 테스트 데이터 정보가 담긴 데이터프레임.
        """
        
        test_info['target'] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        test_info.to_csv(f"/data/ephemeral/home/results/{self.model.model_name}/output.csv", index=False)

    def run_inference(self):
        
        """
        전체 추론 과정을 실행하는 메서드.
        1. 데이터를 준비하고,
        2. 모델을 로드하고,
        3. 추론을 수행한 후,
        4. 결과를 저장합니다.
        """
        
        test_info = self.prepare_data()
        self.prepare_model()
        predictions = self.inference()
        self.save_predictions(predictions, test_info)
