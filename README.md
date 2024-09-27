# Sketch 이미지 데이터 분류 경진대회
<p align="center">
<img src="https://github.com/user-attachments/assets/0c6e68c2-e624-4f81-ab6b-268a19843d24" width="90%" height="90%"/>
</p>

## 1. Competiton Info

### Overview
- ImageNet-Sketch 데이터셋 분류 대회

- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회

- 선정한 상위 500개 객체, 총 25,035개의 ImageNet Sketch 데이터셋을 사용한다 

- 스케치 데이터는 일반 사진과 다르게 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춘다

- 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하여 일반적인 이미지 데이터와의 차이점을 이해한다

### Timeline
2024.09.10 ~ 2024.09.26

### Evaluation
- 제공된 테스트 세트의 각 이미지에 대해 올바른 레이블을 예측해야 한다
- 평가지표: 정확도(Accuracy)

$Accuracy = \dfrac{\text{모델이 올바르게 예측한 샘플 수}}{\text{전체 샘플 수}}$

## 2. Team Info

### MEMBERS

| <img src="https://github.com/user-attachments/assets/b1591118-babe-4de5-9242-793d265edaa4" width="200" height="150"/> | <img src="https://github.com/user-attachments/assets/ac22262e-0922-4dfe-876f-8a3ec2a1f25a" width="200" height="150"/> | <img src="https://github.com/user-attachments/assets/970d2e93-5b3a-4721-9bbd-5bab288e4644" width="200" height="150"/> | <img src="https://github.com/user-attachments/assets/91875b0b-a147-4115-aa41-785f1a351820" width="200" height="150"/> | <img src="https://github.com/user-attachments/assets/51f2b716-ed28-4731-ac3b-c32973f83728" width="200" height="150"/> |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김예진](https://github.com/yeyechu)            |            [배형준](https://github.com/BaeHyungJoon)             |            [송재현](https://github.com/mongsam2)             |            [이재효](https://github.com/jxxhyo)             |            [차성연](https://github.com/MICHAA4)             |

### Project Objective and Direction

- 특정한 방향 없이 자유롭게 탐색하되, 기본에 충실하기
    - 처음 참가하는 대회이기 때문에 전반적인 프로세스를 이해하고 기본 지식을 습득하고자 하였다
    - 개발 파트 구분없이 최대한 많은 것을 경험하고 다양한 인사이트를 도출하고자 하였다
- 실험에 대한 경험치 쌓기
    - 데이터셋에 대해 충분히 이해하고 데이터 augmentation에 대한 다양한 실험을 목표로 하였다
    - 선정한 backbone 기반 모델에 대해 다양한 하이퍼파라미터를 적용해보고자 하였다
- 효율적인 체계 구축과 의사소통
    - 클린 코드를 지향하는 공통의 학습 템플릿 설계를 목표로 하였다
    - 실험 기록 공유, 체계 구축 등 효율적인 커뮤니케이션 방법을 모색하였다

### Team Component
- 김예진 : 데이터 증강 실험(RGB와 Grayscale), 모델 설계 및 실험, 하이퍼파라미터 실험, 실험 관리 툴 제작
- 배형준 : 베이스라인 코드 설계(without_val), 모델 설계 및 실험, 학습 체계 구축
- 송재현 : 데이터 증강 실험(Cutmix), 베이스라인 코드 설계(Config), 모델 설계 및 실험
- 이재효 : 베이스라인 코드 설계(전체, Ensemble), 모델 설계 및 실험, 하이퍼파라미터 실험
- 차성연 : 데이터 증강 실험(워터마크, 문자 제거), 베이스라인 코드 설계(전체, Ensemble), 모델 설계 및 실험, 실험 관리 툴 제작

## 3. Data Description
- 주로 사람의 손으로 그려진 드로잉이나 스케치로 구성되어 있다
- 객체, 동물, 인물, 풍경 등 다양한 카테고리에 걸친 추상적이고 간소화된 이미지들로 이루어져 있다
- 색상이나 세부적인 질감 없이 기본적인 윤곽선과 형태로 표현된다
- 같은 객체를 나타내는 스케치라도 다양한 시각과 표현 스타일을 반영하기에 그린 사람에 따라 매우 다를 수 있다

### Data Tree

```plaintext
data/
│
├── sample_submission.csv - 제출용 csv 파일
├── test.csv - 테스트용 스케치 이미지 
├── train.csv - 학습용 스케치 이미지 
│
├── test/ - 10,014개
│   ├── 0.JPEG
│   ├── 1.JPEG
│   ├── 2.JPEG
│   ├── ...
│
├── train/ - 15,021개
│   ├── n01443537/
│   ├── n01484850/
│   ├── ... 
```

- 500개의 클래스가 존재한다
- 모든 이미지는 JPEG 형식이며, 크기는 서로 다를 수 있지만 대부분 흑백 스케치 이미지로 구성되어 있다
- 클래스별로 약 29~31개의 스케치, 이미지를 포함하고 있어, 데이터 불균형 문제는 거의 존재하지 않는다
- 이미지 크기가 대부분 400px~800px 사이에 분포하여, 이미지 크기 정규화가 필요하다
- 스케치 이미지라 RGB 값이 200에서 250사이에 몰려 있고, 색상 정보를 기반으로 분류하는 것은 크게 중요하지 않을 수 있다
- 대신, 형태 정보에 더 중점을 둔 모델을 설계하는 것이 적합할 수 있다

## 4. Modeling

### Model descrition

| beitv2_large_patch16_224 | deit_base_distilled_patch16_384 |
| --- | --- |
| eva02_base_patch14_448 | eva02_large_patch14_448 |

### Modeling Result

| Model | Accuracy |
| --- | --- |
| Ensemble | 0.9300 |
| eva02_base_patch14_448 | 0.9160 |

## 5. Result

### Leader Board
Team name : CV_22조
<p align="center">
<img src="https://github.com/user-attachments/assets/d6e20035-ff0b-4aae-9677-f04965d45233" width="70%" height="70%"/>
</p>

### Areas for improvement
- 워터마크, 문자 제거를 적용해서 모델 성능을 측정해 보기
- gradient accumulation을 통해 더 큰 batch size로 학습시키기
- sketch image의 위치를 detection해서 그 부분만 crop해서 학습에 사용할 수 있도록 개선하기

---
