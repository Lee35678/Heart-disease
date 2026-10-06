# Heart-disease

NumPy만으로 PCA 특징 추출과 2층 신경망(오차 역전파)을 직접 구현해 심장질환 여부를 분류한 과제 코드.

## 개요

`heart_disease_new.csv` 데이터를 전처리하고, PCA로 4개의 특징을 뽑은 뒤
은닉층 1개짜리 신경망을 sklearn·딥러닝 프레임워크 없이 NumPy로 구현해 학습한다.
학습 과정의 MSE·정확도 곡선과 혼동 행렬(Confusion Matrix)을 matplotlib으로 그린다.

## 주요 내용

1. **전처리** (`select_features`)
   - 결측치를 열 평균으로 대체
   - `gender`(male → 1), `a neurological disorder`(yes → 1), `heart disease`(yes → 1)를 0/1로 변환
   - 정답 열을 제외한 모든 열을 Min-Max 정규화(0~1)
2. **특징 추출 (PCA 직접 구현)** — 평균 중심화 → 공분산 행렬 → `np.linalg.eigh`로 고유값 분해 → 고유값 상위 4개 주성분으로 투영
3. **클래스 불균형 처리** (`oversample_data`) — 소수 클래스를 복원 추출로 늘려 1:1 비율로 맞춤
4. **2층 신경망** (`Two_Layer_Neural_Network`, `Eror_Back_Propagation`)
   - 은닉층·출력층 모두 시그모이드, 은닉층 출력에 바이어스 추가
   - 샘플 단위(온라인) 경사하강 오차 역전파, 손실은 MSE
   - 하이퍼파라미터: 은닉 노드 3, 에폭 1000, 학습률 0.0015
5. **시각화** — 에폭별 MSE / 정확도 그래프, 0.5 기준 이진 판정 혼동 행렬

## 기술 스택

Python, NumPy, pandas, matplotlib

## 실행 방법

```bash
pip install numpy pandas matplotlib
python 2020146024_feature_final.py
```

데이터 파일 경로가 코드에 Windows 다운로드 폴더 경로로 직접 적혀 있다
(`select_features("C:\\Users\\...\\Downloads\\")`). 실행 전에 이 인자를
`heart_disease_new.csv`가 있는 폴더 경로(끝에 경로 구분자 포함)로 바꿔야 한다.

## 폴더 구조

```
Heart-disease/
└── 2020146024_feature_final.py   # 전처리 + PCA + 신경망 학습 + 시각화 전체
```

## 참고

- 데이터셋(`heart_disease_new.csv`)은 저장소에 포함되어 있지 않다.
- 학습/테스트 분리가 없다. 출력되는 "트레이닝 정확도"는 오버샘플링한 학습 데이터 기준이고, 혼동 행렬도 학습에 쓴 원본 데이터로 계산하므로 일반화 성능을 뜻하지 않는다.
- 입력층에는 바이어스 열이 붙지 않는다(주석에는 "바이어스 포함"으로 적혀 있지만 실제로는 특징 4개만 들어간다).
- 난수 시드를 고정하지 않아 실행할 때마다 결과가 달라진다.
- 가중치를 CSV로 저장하는 코드는 주석 처리되어 있다.
