import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

# SSL 인증서 검증 오류를 무시하도록 우회 설정합니다.
ssl._create_default_https_context = ssl._create_unverified_context

st.set_page_config(
    page_title="ML_02_Polynomial_Regression",
    layout="wide"
)

# Matplotlib 한글 폰트 설정 (Windows 기준: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.title("다항 회귀와 과적합: 특성 공학 및 Ridge 규제")
st.write("다중 회귀 모델의 특성 차수를 높여 과적합을 경험하고, StandardScaler와 Ridge 규제를 활용해 예측 성능을 극복합니다.")

# 1. 데이터 로드 및 준비
@st.cache_data
def load_data():
    # 농어 특성 데이터 (길이, 높이, 너비)
    perch_full = pd.read_csv('https://bit.ly/perch_csv_data')
    # 농어 타깃 데이터 (무게)
    perch_weight = np.array([
        5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
        110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
        130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
        197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
        514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
        820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
        1000.0, 1000.0
    ])
    return perch_full, perch_weight

perch_full, perch_weight = load_data()

st.write("### 1. 다중 회귀 데이터셋 확인")
st.write("이 실습에서는 농어의 세 가지 특성인 길이(length), 높이(height), 너비(width)를 사용하여 무게를 예측합니다.")

col_df, col_shape = st.columns([3, 1])
with col_df:
    st.dataframe(perch_full.head(), use_container_width=True)
with col_shape:
    st.metric(label="전체 샘플 수 및 피처 수", value=f"{perch_full.shape[0]}개 x {perch_full.shape[1]}개")

# 훈련/테스트 세트 분리
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42
)

st.divider()

st.write("### 2. 차수(Degree) 변경에 따른 과적합 체험")
st.write("다항 특성의 차수를 변경하여 피처 수의 변화와 모델 학습 점수(R2)의 변화를 실시간으로 확인하세요.")

degree = st.slider("다항식의 차수 (Polynomial Degree)", min_value=1, max_value=5, value=5, step=1)

# PolynomialFeatures 모델 적용
poly = PolynomialFeatures(degree=degree, include_bias=False)
poly.fit(train_input) # 훈련 데이터만 fit
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input) # 테스트 데이터는 transform만 적용

# 선형 회귀 모델 훈련
lr = LinearRegression()
lr.fit(train_poly, train_target)

train_score_lr = lr.score(train_poly, train_target)
test_score_lr = lr.score(test_poly, test_target)

st.code(f"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 차수 지정: degree={degree}
poly = PolynomialFeatures(degree={degree}, include_bias=False)

# 훈련 세트로 다항 특성을 학습(fit)하고 변환(transform)합니다.
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# 선형 회귀 모델 학습
lr = LinearRegression()
lr.fit(train_poly, train_target)

print("훈련 세트 R2:", lr.score(train_poly, train_target))
print("테스트 세트 R2:", lr.score(test_poly, test_target))
""", language="python")

col_res1, col_res2, col_res3 = st.columns(3)
with col_res1:
    st.metric(label="생성된 특성 개수", value=f"{train_poly.shape[1]}개")
with col_res2:
    st.metric(label="일반 선형 회귀 훈련 R2", value=f"{train_score_lr:.4f}")
with col_res3:
    st.metric(label="일반 선형 회귀 테스트 R2", value=f"{test_score_lr:.4f}")

# 과적합 진단 경고
if degree == 5:
    st.error("경고: 특성 개수(55개)가 훈련 데이터 개수(42개)보다 많아 모델이 훈련 데이터를 완벽하게 암기했습니다. 이로 인해 테스트 R2 점수가 극심한 음수(약 -144.4)로 떨어지는 전형적인 과적합(Overfitting) 재앙 상태입니다.")
elif degree >= 3:
    st.warning("경고: 차수가 늘어남에 따라 복잡도가 상승하여 과적합 경향이 강해집니다.")

st.divider()

st.write("### 3. StandardScaler 표준화 및 Ridge 규제 적용")
st.write("과적합 상태인 degree=5 모델에 대해 가중치를 제어하여 일반화 능력을 복원하는 과정을 실습합니다.")

st.write("#### 1단계: StandardScaler를 활용한 변환 (fit 분리 준수)")
st.code("""
from sklearn.preprocessing import StandardScaler

# StandardScaler를 생성하여 다항 피처를 표준 정규분포 척도로 통일합니다.
ss = StandardScaler()
train_scaled = ss.fit_transform(train_poly) # 훈련 데이터셋으로만 fit 및 transform 수행
test_scaled = ss.transform(test_poly)       # 테스트 데이터셋은 transform만 수행 (fit 방지)
""", language="python")

# 스케일러 생성 및 데이터 표준 정규분포화
ss_p = StandardScaler()
train_scaled = ss_p.fit_transform(train_poly)
test_scaled = ss_p.transform(test_poly)

st.write("#### 2단계: Ridge(L2) 규제 적용")
alpha_val = st.select_slider(
    "Ridge 규제 강도 설정 (alpha)", 
    options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 
    value=0.1
)

# Ridge 모델 훈련
ridge = Ridge(alpha=alpha_val)
ridge.fit(train_scaled, train_target)

train_score_ridge = ridge.score(train_scaled, train_target)
test_score_ridge = ridge.score(test_scaled, test_target)

st.code(f"""
from sklearn.linear_model import Ridge

# 규제 강도: alpha={alpha_val}
ridge = Ridge(alpha={alpha_val})
ridge.fit(train_scaled, train_target)

print("Ridge 훈련 R2:", ridge.score(train_scaled, train_target))
print("Ridge 테스트 R2:", ridge.score(test_scaled, test_target))
""", language="python")

col_ridge1, col_ridge2 = st.columns(2)
with col_ridge1:
    st.metric(label="Ridge 규제 훈련 세트 R2", value=f"{train_score_ridge:.4f}")
with col_ridge2:
    st.metric(label="Ridge 규제 테스트 세트 R2", value=f"{test_score_ridge:.4f}")

if alpha_val == 0.1:
    st.info("알림: alpha = 0.1 설정 시 규제가 균형 있게 적용되어 훈련 R2 점수와 테스트 R2 점수 간 격차가 좁혀지고, 테스트 R2 점수가 최고 수준(약 0.98)으로 복원됩니다.")

st.divider()

st.write("### 4. degree=2 결과와 degree=5 결과 테이블 비교")
st.write("교재에 제시된 농어 다중 회귀 모델의 대표 차수인 2차와 5차에 대하여 일반 선형 모델과 Ridge 규제 모델의 지표를 요약하여 비교합니다.")

# degree 2 연산
poly2 = PolynomialFeatures(degree=2, include_bias=False)
train_poly2 = poly2.fit_transform(train_input)
test_poly2 = poly2.transform(test_input)

lr2 = LinearRegression()
lr2.fit(train_poly2, train_target)

ss2 = StandardScaler()
train_scaled2 = ss2.fit_transform(train_poly2)
test_scaled2 = ss2.transform(test_poly2)

ridge2 = Ridge(alpha=0.1)
ridge2.fit(train_scaled2, train_target)

# degree 5 연산 (슬라이더 상관없이 고정 연산)
poly5 = PolynomialFeatures(degree=5, include_bias=False)
train_poly5 = poly5.fit_transform(train_input)
test_poly5 = poly5.transform(test_input)

lr5 = LinearRegression()
lr5.fit(train_poly5, train_target)

ss5 = StandardScaler()
train_scaled5 = ss5.fit_transform(train_poly5)
test_scaled5 = ss5.transform(test_poly5)

ridge5 = Ridge(alpha=0.1)
ridge5.fit(train_scaled5, train_target)

# 비교 데이터프레임 구축
summary_data = {
    "구분 항목": [
        "생성된 다항 피처 수", 
        "일반 선형 모델 훈련 R2", 
        "일반 선형 모델 테스트 R2", 
        "선형 모델 최종 진단",
        "Ridge(alpha=0.1) 훈련 R2", 
        "Ridge(alpha=0.1) 테스트 R2", 
        "Ridge 규제 모델 최종 진단"
    ],
    "degree=2 (곡선 피팅)": [
        f"{train_poly2.shape[1]}개",
        f"{lr2.score(train_poly2, train_target):.4f}",
        f"{lr2.score(test_poly2, test_target):.4f}",
        "안정적 (적절한 곡선 피팅)",
        f"{ridge2.score(train_scaled2, train_target):.4f}",
        f"{ridge2.score(test_scaled2, test_target):.4f}",
        "우수함"
    ],
    "degree=5 (특성 과적합)": [
        f"{train_poly5.shape[1]}개",
        f"{lr5.score(train_poly5, train_target):.4f}",
        f"{lr5.score(test_poly5, test_target):.4f}",
        "심각한 과적합 (훈련 데이터 완벽 암기)",
        f"{ridge5.score(train_scaled5, train_target):.4f}",
        f"{ridge5.score(test_scaled5, test_target):.4f}",
        "우수함 (테스트 R2가 정상 회복됨)"
    ]
}

df_summary = pd.DataFrame(summary_data)
st.table(df_summary)

st.divider()

st.write("### 5. 가중치(coef_) 절댓값 크기 변화 비교")
st.write("일반 선형 회귀 모델과 Ridge 규제 모델의 가중치 변동성을 시각화하여, 규제가 가중치 폭증을 어떻게 강제로 줄여주는지 직관적으로 규명합니다.")

# degree=5 모델의 가중치 정보 획득
lr_coefs = lr5.coef_
ridge_coefs = ridge5.coef_

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np.abs(lr_coefs), label="일반 선형 모델 가중치 절댓값", color="#EB5757", marker="x", alpha=0.8)
ax.plot(np.abs(ridge_coefs), label="Ridge 규제 가중치 절댓값 (alpha=0.1)", color="#2F80ED", marker="o", alpha=0.8)

ax.set_title("일반 선형 회귀 vs Ridge 가중치 변동성 비교 (degree=5)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("가중치 인덱스 (총 55개)", fontsize=12)
ax.set_ylabel("가중치 절댓값 (로그 스케일)", fontsize=12)
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()
st.pyplot(fig)
