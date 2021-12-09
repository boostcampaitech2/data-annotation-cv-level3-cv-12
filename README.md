# data-annotation-cv-level3-cv-12

## 파일- camp 제공 x
concat_json.py : 여러 출처의 데이터 파일들을 한 개의 json 으로 처리하기 위한 파일
convert_json.py : 현재 갖고 있는 문제점들을 제거하기 위해 넣은 파일

## 팀 회고록
글자 검출 대회 Wrap-up Report.pdf

## 파일 - camp 제공 o 
data_augmentation.ipynb : data augmentation을 어떻게 할 수 있는지 정리되어 있는 파일 from office hour
dataset.py: 학습에 필요한 데이터셋이 정의되어 있고 augmentation 또한 정리되어 있는 파일
detect.py : 모델의 추론에 필요한 기타 함수들이 정의되어 있는 파일
deteavl.py : DetEval 평가를 위한 함수들이 정의되어 있는 파일
east_atoz.ipynb : EAST 학습에 필요한 데이터셋이 어떤 과정을 거쳐 train에 입력되는지를 보여주는 파일 from office hour
(east_dataset.py : EAST 학습에 필요한 형식의 데이터셋이 정의되어 있는 파일, east_atoz.ipynb에서 east_dataset에서 하는 일들이 잘 정리되어 있다.)
inference.py : 모델의 추론 절차가 정의되어 있는 파일
loss.py : 학습을 위한 loss fucntion이 정의되어 있는 파일입니다
model.py: EAST 모델이 정의된 파일(vgg16 사용)
OCR_EDA.ipynb : Dataset에 대한 eda 파일
train.py : 모델 학습 절차가 정의되어 있는 파일

후에 data_augmentation.ipynb, east_atoz.ipynb 를 조금 더 눈여겨봄직하다.
