import os
import timm
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset import CustomDataset
from model.model_selector import ModelSelector
from transform.transform_selector import TransformSelector

class ModelEnsemble:
    
    """
    여러 모델을 사용하여 앙상블 추론을 수행하는 클래스입니다.

    Args:
        testdata_dir (str): 테스트 데이터가 위치한 디렉토리 경로.
        testdata_info_file (str): 테스트 데이터에 대한 정보가 저장된 CSV 파일 경로.
        save_result_path (str): 예측 결과를 저장할 경로.
        model_configs (list): 사용할 모델들의 이름과 입력 이미지 크기가 포함된 리스트.
        batch_size (int): 배치 크기.
        num_classes (int): 예측할 클래스의 수.
        pretrained (bool): 사전 학습된 모델을 사용할지 여부.
        num_workers (int): 데이터 로딩에 사용할 워커(worker)의 수.
    """
    
    def __init__(self, 
                 testdata_dir, 
                 testdata_info_file, 
                 save_result_path, 
                 model_configs=[{'name':'beitv2_large_patch16_224', 'input_size':(224, 224)},
                                {'name':'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384', 'input_size':(384, 384)}],
                 batch_size=64,
                 num_classes=500, 
                 pretrained=False,
                 num_workers=6):
        
        """
        ModelEnsemble 클래스의 생성자 함수입니다. 초기 변수들을 설정하고, 장치를 설정합니다.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.testdata_dir = testdata_dir
        self.testdata_info_file = testdata_info_file
        self.save_result_path = save_result_path
        self.model_configs = model_configs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_workers = num_workers
        self.models = []
        self.transforms = []
        self.test_loader = None

    def prepare_data(self):
        
        """
        테스트 데이터를 준비하고, DataLoader를 설정하는 함수입니다.

        Returns:
            test_info (pandas.DataFrame): 테스트 데이터에 대한 정보가 포함된 데이터프레임.
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
        설정된 모델 구성에 따라 모델을 로드하고, 각 모델별 전용 전처리(transform)를 설정하는 함수입니다.
        """
        
        for config in self.model_configs:
            model_name = config['name']
            input_size = config['input_size']
            
            if model_name.endswith('_origin') or model_name.endswith('_cutmix')  or model_name.endswith('_cutmix_train') or model_name.endswith('_train'):
                model_change_name = model_name.replace('_origin', '').replace('_cutmix', '').replace('_cutmix_train', '').replace('_train', '')
            else:
                model_change_name = model_name  
        
            model_selector = ModelSelector(
                model_type='timm', 
                num_classes=self.num_classes,
                model_name=model_change_name, 
                pretrained=self.pretrained
            )
            model = model_selector.get_model()

            model_path = os.path.join(self.save_result_path, model_name, "best_model.pt")
            model.load_state_dict(torch.load(model_path, map_location='cpu'))

            model.to(self.device)
            model.eval()

            self.models.append(model)

            model_transform = transforms.Compose([
                transforms.Resize(input_size),
            ])
            self.transforms.append(model_transform)

    def inference(self):
        
        """
        준비된 모델들을 사용하여 테스트 데이터를 기반으로 추론을 수행하는 함수입니다.

        Returns:
            predictions (list): 각 이미지에 대한 최종 예측 클래스의 리스트.
        """
        
        predictions = []
        with torch.no_grad():
            for images in tqdm(self.test_loader):
                ensemble_preds = []

                for i, model in enumerate(self.models):
                    model_transform = self.transforms[i]

                    resized_images = torch.stack([model_transform(image) for image in images])
                    resized_images = resized_images.to(self.device)

                    with torch.cuda.amp.autocast():
                        logits = model(resized_images)
                    preds = logits.argmax(dim=1)
                    ensemble_preds.append(preds.cpu().detach().numpy())

                ensemble_preds = np.array(ensemble_preds)
                final_preds = []
                for i in range(ensemble_preds.shape[1]):
                    sample_preds = ensemble_preds[:, i]
                    final_pred = np.bincount(sample_preds).argmax()
                    final_preds.append(final_pred)
                
                predictions.extend(final_preds)

        return predictions

    def save_predictions(self, predictions, test_info):
        
        """
        예측 결과를 저장하는 함수입니다.

        Args:
            predictions (list): 예측된 클래스 리스트.
            test_info (pandas.DataFrame): 테스트 데이터에 대한 정보가 포함된 데이터프레임.
        """
        
        test_info['target'] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        
        output_dir = os.path.join(self.save_result_path, "ensemble")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "output.csv")
        test_info.to_csv(output_file, index=False)

    def run_inference(self):
        
        """
        전체 추론 과정을 실행하는 함수입니다. 데이터 준비, 모델 준비, 추론 및 결과 저장을 포함합니다.
        """
        
        test_info = self.prepare_data()
        self.prepare_model()
        predictions = self.inference()
        self.save_predictions(predictions, test_info)
        