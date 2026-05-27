import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="ML_02_k-NN_Linear_Regression",
    layout="wide"
)

# Matplotlib 한글 폰트 설정 (Windows 기준: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.title("농어 무게 예측: k-NN 회귀 + 선형 회귀")
st.write("k-NN 회귀와 선형 회귀 모델을 각각 훈련하고 성능을 비교합니다.")

# 농어 데이터 정의
perch_length = np.array([
    8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
    21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
    22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
    27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
    36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
    40.0, 42.0, 43.0, 43.0, 43.5, 44.0
])

perch_weight = np.array([
    5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
    110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
    130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
    197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
    514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
    820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
    1000.0, 1000.0
])

st.write("### 1. 데이터 준비 및 전처리")
st.write("사이킷런을 활용해 학습하기 위해 perch_length에 2차원 형상 변환인 reshape(-1, 1)을 적용하여 데이터를 2차원 열 벡터 구조로 구축합니다.")

st.code("""
# numpy 배열을 2차원 형태로 변환합니다.
train_input = perch_length.reshape(-1, 1)
train_target = perch_weight
""", language="python")

# reshape(-1, 1) 적용
X = perch_length.reshape(-1, 1)
y = perch_weight

# 훈련/테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

st.write("변환 완료된 입력 데이터 형상 (X_train.shape):", X_train.shape)

st.divider()

st.write("### 2. k-NN 회귀 모델 학습")
st.write("KNeighborsRegressor 모델을 객체화하고 훈련 데이터에 피팅하여 성능 점수를 출력합니다.")

# KNeighborsRegressor().fit(train, target) 실행
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(X_train, y_train)

# R2 점수 계산
train_score_knn = knr.score(X_train, y_train)
test_score_knn = knr.score(X_test, y_test)

st.code("""
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(X_train, y_train)

# 성능(결정 계수) 점수 출력
print("train R2:", knr.score(X_train, y_train))
print("test R2:", knr.score(X_test, y_test))
""", language="python")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="k-NN 훈련 세트 R2 점수", value=f"{train_score_knn:.4f}")
with col2:
    st.metric(label="k-NN 테스트 세트 R2 점수", value=f"{test_score_knn:.4f}")

st.divider()

st.write("### 3. 선형 회귀 모델 학습")
st.write("LinearRegression 모델을 학습하고 가중치(lr.coef_)와 편향(lr.intercept_)을 출력합니다.")

# LinearRegression().fit(train, target) 실행
lr = LinearRegression()
lr.fit(X_train, y_train)

# R2 점수 계산
train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

st.code("""
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

# 가중치 및 편향 파라미터 값 출력
print("coef_:", lr.coef_)
print("intercept_:", lr.intercept_)
""", language="python")

col3, col4 = st.columns(2)
with col3:
    st.metric(label="선형 회귀 가중치 (lr.coef_)", value=f"{lr.coef_[0]:.4f}")
    st.metric(label="선형 회귀 훈련 세트 R2 점수", value=f"{train_score_lr:.4f}")
with col4:
    st.metric(label="선형 회귀 편향 (lr.intercept_)", value=f"{lr.intercept_:.4f}")
    st.metric(label="선형 회귀 테스트 세트 R2 점수", value=f"{test_score_lr:.4f}")

st.divider()

st.write("### 4. 모델 예측 결과 및 한계 시각화 비교")
st.write("슬라이더를 사용해 새로운 길이를 입력하고 두 모델의 예측 무게와 외삽 동작 방식의 물리적 차이를 그래프로 비교해 보세요.")
st.write("45cm 부터 KNN 훈련 범위를 벗어나서 항상 고정된 최대 평균치를 반환합니다. (= 외삽 학습 불가 문제)")

input_len = st.slider("농어의 길이 입력 (cm)", min_value=10.0, max_value=100.0, value=50.0, step=1.0)

# 예측 수행
pred_knn = knr.predict([[input_len]])[0]
pred_lr = lr.predict([[input_len]])[0]

col5, col6 = st.columns(2)
with col5:
    st.write(f"k-NN 회귀 예측 무게: **{pred_knn:.2f}g**")
    if input_len > 44.0:
        st.warning("k-NN 회귀 한계: 훈련 범위를 벗어나 항상 고정된 최대 평균치를 반환합니다. (외삽 불가)")
with col6:
    st.write(f"선형 회귀 예측 무게: **{pred_lr:.2f}g**")
    if input_len < 18.0:
        st.warning("선형 회귀 한계: 짧은 농어 입력 시 비현실적인 음수의 무게를 도출할 위험이 있습니다.")

# 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 6))

# 산점도 플로팅
ax.scatter(X_train, y_train, label="훈련 세트", color="#2F80ED", alpha=0.7, edgecolors="k", s=60)
ax.scatter(X_test, y_test, label="테스트 세트", color="#F2994A", alpha=0.7, edgecolors="k", s=60)

# 모델 예측 곡선 그리기
x_range = np.arange(10, 100).reshape(-1, 1)
y_knn_pred = knr.predict(x_range)
y_lr_pred = lr.predict(x_range)

ax.plot(x_range, y_knn_pred, label="k-NN 회귀 예측선", color="#27AE60", linestyle="--", linewidth=2)
ax.plot(x_range, y_lr_pred, label="선형 회귀 예측선", color="#EB5757", linewidth=2)

# 현재 시뮬레이션 지점 표시
ax.scatter([input_len], [pred_knn], color="#27AE60", marker="X", s=200, edgecolors="k", zorder=5, label=f"k-NN 현재 예측 ({pred_knn:.1f}g)")
ax.scatter([input_len], [pred_lr], color="#EB5757", marker="o", s=200, edgecolors="k", zorder=5, label=f"선형 회귀 현재 예측 ({pred_lr:.1f}g)")

ax.set_title("k-NN 회귀와 선형 회귀 예측 곡선 비교 분석", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("길이 (cm)", fontsize=12)
ax.set_ylabel("무게 (g)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper left")

plt.tight_layout()
st.pyplot(fig)
