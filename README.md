# Sketch 이미지 데이터 분류 경진대회
![image](https://github.com/user-attachments/assets/0c6e68c2-e624-4f81-ab6b-268a19843d24)


- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회
- Sketch Image Data Classification


## MEMBERS

| ![박패캠](https://cdn.bootcampkorea.com/bootcamp/Boostcamp.png) | ![이패캠](https://cdn.bootcampkorea.com/bootcamp/Boostcamp.png) | ![최패캠](https://cdn.bootcampkorea.com/bootcamp/Boostcamp.png) | ![김패캠](https://cdn.bootcampkorea.com/bootcamp/Boostcamp.png) | ![오패캠](https://cdn.bootcampkorea.com/bootcamp/Boostcamp.png) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김예진](https://github.com/yeyechu)             |            [배형준](https://github.com/BaeHyungJoon)             |            [송재현](https://github.com/mongsam2)             |            [이재효](https://github.com/jxxhyo)             |            [차성연](https://github.com/MICHAA4)             |
|                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 1. Competiton Info

### Overview
- 주어진 데이터를 활용하여 이미지가 어떤 라벨을 나타내는지 분류하는 모델을 개발하는 대회이다
- 스케치 데이터는 일반 사진과 다르게 색상, 질감, 세부적인 형태가 비교적 결여되어 있으며, 대신에 기본적인 형태와 구조에 초점을 맞춘다
- 스케치 데이터의 특성을 이해하고 스케치 이미지를 통해 모델이 객체의 기본적인 형태와 구조를 학습하여 일반적인 이미지 데이터와의 차이점을 이해한다
- 선정한 상위 500개 객체, 총 25,035개의 ImageNet Sketch 데이터셋을 사용한다
    

### Timeline
2024.09.10 ~ 2024.09.26

### Evaluation
- 손으로 그린 스케치 이미지가 어떤 객체(class)를 나타내는지 분류하는 대회입니다.
- 제공된 테스트 세트의 각 이미지에 대해 올바른 레이블을 예측해야 합니다.
- 평가지표는 정확도(Accuracy)를 사용합니다.


## 2. Team Component
- 김예진 : 모델 설계 및 실험(cat, xgboost,lightGBM), optuna로 파라미터 튜닝 
- 배형준 : geocoding을 이용한 좌표X와 좌표Y의 결측치 보완 및 feature 생성
- 이재효 : 전반적인 전처리 및 feature importance 찾기
- 차성연 : 범주형 데이터 인코딩(one-hot, label) 및 k-fold수행
- 송재현 : 팀의 기본적인 목표 설정, 협업 보조 및 공부

## 3. Data Descrption
- 주로 사람의 손으로 그려진 드로잉이나 스케치로 구성되어 있습니다.
- 객체, 동물, 인물, 풍경 등 다양한 카테고리에 걸친 추상적이고 간소화된 이미지들로 이루어져 있습니다.
- 색상이나 세부적인 질감 없이 기본적인 윤곽선과 형태로 표현됩니다.
- 같은 객체를 나타내는 스케치라도 다양한 시각과 표현 스타일을 반영하기에 그린 사람에 따라 매우 다를 수 있습니다.
### Dataset overview
![image](https://github.com/user-attachments/assets/90e90749-7bbb-4ce3-8887-448a27a13c3f)

#### - 데이터 구성

```plaintext
data/
│
├── sample_submission.csv 
├── test.csv - 스케치 이미지 
├── train.csv - 스케치 이미지 
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
###  EDA


### Feature engineering
1. 강남/서초/용산구
   평균 실거래가가 높은 구는 강남구, 서초구, 용산구 -> 3개의 구에 해당하면 1, 아니면 0으로 처리
2. 건설사
   345개의 unique한 건설사 이름 존재 -> 특정 회사 단어를 기준으로 묶어 범주화 함 (345->136)
3. 버스, 지하철
   버스, 지하철 geocoding으로 주소정보를 이용하여 결측치 보완, 인접 역 정보(거리, 이름) 생성
4. 전용면적
   90 이하일 경우 1, 초과일경우 0
5. 아파트명 정리
   정규표현식을 이용 cardinality를 줄임 : n동, n차, n단지 등 제거(6586 -> 4362)
6. 아파트 브랜드
   아파트 브랜드가 부동산 가격에 영향을 미침 : 브랜드 있는 아파트명 1, 아니면 0으로 처리
   
## 4. Modeling

Encoding(one-hot, label) : 각 변수마다 적절한 인코딩을 사용

### Model descrition

사용 모델(CatBoost, XGBoost, LightGBM) 

### Modeling Result
<img width="799" alt="스크린샷 2024-04-02 오후 4 17 06" src="https://github.com/UpstageAILab2/upstage-ml-regression-ml-01/assets/156395327/9f05d5c4-bbfd-4a7e-8f52-7ea3f5786c92">


## 5. Result

### Leader Board
Team name : ML_1조
<img width="1072" alt="스크린샷 2024-04-02 오후 4 22 07" src="https://github.com/UpstageAILab2/upstage-ml-regression-ml-01/assets/156395327/09f4dde0-5359-4734-b5d1-8696dbfca0e1">

### Presentation
[ML 1조 발표.pptx.pdf](https://github.com/UpstageAILab2/upstage-ml-regression-ml-01/files/14833025/ML.1.pptx.pdf)

### Areas for improvement
- 서버에서 저장한 CatBoost와 LGBM이 로컬에서 돌아가지 않아 앙상블을 수행하지 못함 ---> 추후에 진행할 예정
- 라우터를 사용하고 하였으나 혼선이 생겨 에러를 해결하지 못함 ---> 추후 라우터 사용을 위한 코드 수정
