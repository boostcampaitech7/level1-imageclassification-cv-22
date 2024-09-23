# Sketch 이미지 데이터 분류
## Project base template

```
baseline_code_template
|-- main.py : 메인 스크립트 및 학습(train)과 추론(inference) 진행
|-- train.py : 모델 학습
|-- inference.py : 학습된 모델로 추론
|-- dataset 
|   -- dataset.py : 데이터셋을 로딩하고 전처리
|-- loss
|   -- loss.py : 손실 함수 정의
|-- model
|   |-- model_selector.py : 다양한 모델 중에서 사용할 모델을 선택
|   |-- simple_cnn.py 
|   |-- timm_model.py
|   |-- torchvision_model.py
|-- trainer
|   -- trainer.py : 모델 학습 과정(training)을 관리
|-- transform
    |-- transform_selector.py : 사용하고자 하는 변환 방식을 선택
    |-- albumentations_transform.py
    |-- torchvision_transform.py
|-- requirements.txt : 프로젝트에서 사용되는 파이썬 패키지 및 의존성 목록
