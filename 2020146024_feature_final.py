# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:51:04 2024

@author: 동혁
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def select_features(directory):
    path = directory + "heart_disease_new.csv"
    df = pd.read_csv(path)
    
    # 데이터 전처리
    # NaN 값을 평균값으로 대체
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # 데이터 형식 변환
    df["gender"] = np.where(df["gender"] == "male", 1, 0)
    df["a neurological disorder"] = np.where(df["a neurological disorder"] == "yes", 1, 0)
    df["heart disease"] = np.where(df["heart disease"] == "yes", 1, 0)
    
    # 정규화
    # y값을 제외하고 기존의 14개의 특징들을 정규화를 통해 0과 1사이의 값들로 변환
    for column in df.columns:
        if column != "heart disease":
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    # 데이터 랜덤으로 섞기
    dataset = np.array(df)
    np.random.shuffle(dataset)
    
    # 특징 추출
    y_data = dataset[:, -1] # y값
    X_data = dataset[:, :-1] # 기존의 특징들
    
    # 평균 중심화
    X_centered = X_data - np.mean(X_data, axis=0)
    # 공분산 행렬 계산
    cov_matrix = np.cov(X_centered, rowvar=False)
    # 고유값과 고유벡터 계산
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # 고유값을 내림차순으로 정렬
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    # 선택된 주성분
    eigenvector_subset = sorted_eigenvectors[:, :4]
    # 데이터 변환
    X_reduced = np.dot(X_centered, eigenvector_subset)
    
    # 새로운 특징 생성
    feature_1 = X_reduced[:, 0] # 특징 1
    feature_2 = X_reduced[:, 1] # 특징 2
    feature_3 = X_reduced[:, 2] # 특징 3
    feature_4 = X_reduced[:, 3] # 특징 4

    # 특징 합치기
    select_features = np.column_stack((feature_1, feature_2, feature_3, feature_4))
    # 특징 데이터 마지막 열에 y값 추가
    features = np.column_stack((select_features, y_data))
    
    return features

# 시그모이드 함수 정의
def sigmoid_Func(z):
    return 1 / (1 + np.exp(-z))

# 시그모이드 함수 미분 정의
def sigmoid_deriv(z):
    return z * (1 - z)

# 2층 신경망 함수(순전파)
def Two_Layer_Neural_Network(X, v, w):
    alpha_L = np.dot(X, v)
    f_alpha = sigmoid_Func(alpha_L)
    f_alpha_bias = np.c_[np.ones(f_alpha.shape[0]), f_alpha]  # Add bias
    beta_Q = np.dot(f_alpha_bias, w)
    y_hat = sigmoid_Func(beta_Q)
    return f_alpha_bias, y_hat

# 정확도 계산 함수
def calculate_accuracy(y_true, y_pred):
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / y_true.shape[0]
    return accuracy

# mse 계산 함수
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 데이터의 불균형 문제를 해결하기 위해 오버샘플링 적용
# (심근경색 여부, y값) : ("yes" : 500, "no" : 2500) -> ("yes" : 2500, "no" : 2500) 으로 비율을 1대 1로 맞춤
def oversample_data(data):
    yes_data = data[data[:, -1] == 1]
    no_data = data[data[:, -1] == 0]
    
    yes_count = yes_data.shape[0]
    no_count = no_data.shape[0]
    
    if yes_count < no_count:
        oversampled_yes_data = yes_data[np.random.choice(yes_count, no_count - yes_count, replace=True)]
        balanced_data = np.vstack((data, oversampled_yes_data))
    elif no_count < yes_count:
        oversampled_no_data = no_data[np.random.choice(no_count, yes_count - no_count, replace=True)]
        balanced_data = np.vstack((data, oversampled_no_data))
    else:
        balanced_data = data
    
    np.random.shuffle(balanced_data)
    return balanced_data

# 오차 역전파 알고리즘
def Eror_Back_Propagation(data, hidden_nodes, epochs, learning_rate):
    x = np.ones((data.shape[0], data.shape[1] - 1))     # 바이어스를 위해 1로 초기화
    x[:, :] = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    
    M = x.shape[1]      # 입력 속성 수 (바이어스 포함)
    L = hidden_nodes    # 히든 레이어의 노드 수
    Q = 1               # 출력 레이어의 클래스 수
    
    # 가중치 행렬 생성 및 랜덤 초기화
    v = np.random.randn(M, L)
    w = np.random.randn(L + 1, Q) 
    
    loss_history = []
    accuracy_history = []
    
    for epoch in range(epochs):
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        for n in range(x.shape[0]):
            xn = x[n, :].reshape(1, -1)
            yn = y[n, :].reshape(1, -1)
            
            f_alpha_bias, yn_hat = Two_Layer_Neural_Network(xn, v, w)
            
            delta_q = (yn_hat - yn) * sigmoid_deriv(yn_hat)
            
            w -= learning_rate * np.dot(f_alpha_bias.T, delta_q)
            
            delta_l = np.dot(delta_q, w[1:, :].T) * sigmoid_deriv(f_alpha_bias[:, 1:])
            
            v -= learning_rate * np.dot(xn.T, delta_l)
        
        # Epoch 마다 Loss와 Accuracy 계산
        _, yn_hat_epoch = Two_Layer_Neural_Network(x, v, w)
        loss = mse(y, yn_hat_epoch)
        accuracy = calculate_accuracy(y, yn_hat_epoch)
        
        loss_history.append(loss)
        accuracy_history.append(accuracy)
    
    return v, w, accuracy, loss_history, accuracy_history, yn_hat_epoch

# 디렉토리에 있는 데이터 로드및 특징 추출
features = select_features("C:\\Users\\동혁\\Downloads\\")

# 오버샘플링 적용
resampled_data = oversample_data(features)

hidden_nodes = 3
epochs = 1000
learning_rate = 0.0015

v, w, training_accuracy, loss_history, accuracy_history, y_pred = Eror_Back_Propagation(resampled_data, hidden_nodes, epochs, learning_rate)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history, label='MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Epoch vs MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy')
plt.legend()

plt.show()

y = features[:, -1].reshape(-1, 1)
_, y_hat = Two_Layer_Neural_Network(features[:, :-1], v, w)

print(f"트레이닝 정확도: {training_accuracy}")

# confusion matrix 생성
y_pred_classes = (y_hat >= 0.5).astype(int)
confusion_matrix = np.zeros((2, 2), dtype=int)
for true, pred in zip(y.astype(int), y_pred_classes):
    confusion_matrix[true[0], pred[0]] += 1

# Confusion matrix 시각화
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
for (i, j), val in np.ndenumerate(confusion_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center')
plt.xlabel('predict')
plt.ylabel('true')
plt.title('Confusion Matrix')
ax.set_xticklabels(['', '0 (no)', '1 (yes)'])
ax.set_yticklabels(['', '0 (no)', '1 (yes)'])
plt.show()

# 가중치 v와 w를 CSV 파일로 저장
v_df = pd.DataFrame(v.T)
w_df = pd.DataFrame(w.T)

# 인덱스를 포함하지 않고 CSV 파일로 저장
#v_df.to_csv("C:\\Users\\동혁\\OneDrive\\바탕 화면\\언어 프로그램\\파이썬\\인공지능\\2020146024\\w_hidden.csv", index=False, header=False)
#w_df.to_csv("C:\\Users\\동혁\\OneDrive\\바탕 화면\\언어 프로그램\\파이썬\\인공지능\\2020146024\\w_output.csv", index=False, header=False)

