import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssl
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline

# SSL 인증 오류 우회 설정
ssl._create_default_https_context = ssl._create_unverified_context

# Streamlit 페이지 설정
st.set_page_config(
    page_title="ML_02_California_Regression",
    layout="wide"
)

# Matplotlib 한글 폰트 설정 (Windows 기준: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.title("🏡 캘리포니아 주택 가격 예측: 다중 회귀 모델 및 규제 탐색")
st.write("캘리포니아 주택 가격 데이터셋을 기반으로 **다중 선형 회귀(Multiple Linear Regression)** 모델을 훈련하고, **L1 규제(Lasso)** 및 **L2 규제(Ridge)**를 적용해 하이퍼파라미터 $\\alpha$에 따른 모델 성능 변화와 특성 중요도를 실시간으로 추적 및 시뮬레이션합니다.")

st.divider()

# 1. 데이터 로드 및 탐색
@st.cache_data
def load_data():
    california = fetch_california_housing(as_frame=True)
    df = california.frame
    feature_names = california.feature_names
    target_name = 'MedHouseVal'
    return california, df, feature_names, target_name

california, df, feature_names, target_name = load_data()

st.write("### 1. 데이터셋 소개 및 탐색")
st.markdown("""
**캘리포니아 주택 가격(California Housing) 데이터셋**은 머신러닝에서 회귀(Regression) 문제를 연습할 때 가장 널리 사용되는 대표적인 데이터셋 중 하나입니다.
이 데이터셋은 **1990년 미국 인구조사(Census)** 데이터를 바탕으로 하며, 캘리포니아의 각 조사 구역(Block Group)별 특징과 주택 가격 중간값을 포함하고 있습니다.
""")

col_info, col_feat = st.columns([1, 1])

with col_info:
    st.info("""
    **📊 데이터셋 구조 및 요약**
    - **샘플 수(행, Instances):** 20,640개
    - **특성 수(열, Attributes):** 8개의 수치형 독립 변수
    - **타깃 변수(Target):** 1개의 연속형 종속 변수 (주택 가격 중간값)
    - **결측치(Missing Values):** 존재하지 않음 (모든 데이터가 깔끔하게 채워져 있습니다.)
    - **타깃 Capping:** 실제 데이터에서는 50만 달러($5.0$) 이상의 주택 가격이 모두 `5.0`으로 상한선 처리되어 있는 특징이 있습니다.
    """)

with col_feat:
    st.markdown("""
    **🔍 독립 변수 (Features, 8개)**
    1. **`MedInc` (Median Income)**: 해당 구역 가구들의 **소득 중간값** *(단위: 1만 달러)*
    2. **`HouseAge` (Median House Age)**: 해당 구역 주택들의 **연식 중간값**
    3. **`AveRooms` (Average Rooms)**: 가구당 **평균 방 개수**
    4. **`AveBedrms` (Average Bedrooms)**: 가구당 **평균 침실 개수**
    5. **`Population` (Population)**: 해당 구역의 **총 인구수**
    6. **`AveOccup` (Average Occupancy)**: 가구당 **평균 구성원 수**
    7. **`Latitude` (Latitude)**: 구역의 **위도** (지리적 위치)
    8. **`Longitude` (Longitude)**: 구역의 **경도** (지리적 위치)
    """)

tab1, tab2 = st.tabs(["📋 데이터 상위 5개 행 미리보기", "📈 기술 통계량 (Descriptive Statistics)"])
with tab1:
    st.dataframe(df.head(), use_container_width=True)
with tab2:
    st.dataframe(df.describe(), use_container_width=True)

# 훈련/테스트 세트 분리
X = df.drop(columns=[target_name])
y = df[target_name]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.divider()

# 2. Ridge & Lasso 규제 alpha 파라미터 탐색
st.write("### 2. 정규화 모델(Ridge & Lasso) alpha 하이퍼파라미터 실시간 탐색")
st.write("특성 스케일러(`StandardScaler`)와 선형 회귀 모델(`Ridge` 또는 `Lasso`)을 연결하는 파이프라인을 실시간으로 구축하고, 규제 강도 $\\alpha$를 변경하면서 모델의 과적합/과소적합 곡선을 모니터링합니다.")

col_ctrl, col_metric = st.columns([1, 1])

with col_ctrl:
    st.write("#### 🛠️ 규제 설정")
    model_type = st.radio("회귀 모델 선택", options=["Ridge (L2 규제)", "Lasso (L1 규제)"], horizontal=True)
    alpha_val = st.select_slider(
        "규제 강도 설정 (alpha)",
        options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        value=10.0
    )

# 전체 alpha에 대해 모델 학습 및 스코어 계산 (R^2 곡선 플로팅 용)
alpha_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
train_scores = []
test_scores = []

for a in alpha_list:
    if "Ridge" in model_type:
        model = Ridge(alpha=a)
    else:
        model = Lasso(alpha=a)
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    train_scores.append(pipeline.score(X_train, y_train))
    test_scores.append(pipeline.score(X_test, y_test))

# 현재 선택된 alpha 값으로 파이프라인 최종 피팅
if "Ridge" in model_type:
    selected_model = Ridge(alpha=alpha_val)
else:
    selected_model = Lasso(alpha=alpha_val)
best_pipeline = make_pipeline(StandardScaler(), selected_model)
best_pipeline.fit(X_train, y_train)
selected_train_score = best_pipeline.score(X_train, y_train)
selected_test_score = best_pipeline.score(X_test, y_test)

with col_metric:
    st.write("#### 🎯 실시간 성능 점수")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label=f"{model_type.split()[0]} 훈련 세트 R²", value=f"{selected_train_score:.5f}")
    with col_m2:
        st.metric(label=f"{model_type.split()[0]} 테스트 세트 R²", value=f"{selected_test_score:.5f}")

    # 성능 갭 및 분석 조언
    score_diff = selected_train_score - selected_test_score
    if score_diff > 0.05:
        st.warning("⚠️ 과적합(Overfitting) 경향이 감지되었습니다! alpha 값을 더 늘려 규제를 강화하는 것을 권장합니다.")
    elif selected_train_score < 0.5:
        st.error("❌ 모델이 데이터의 경향을 충분히 설명하지 못하는 과소적합(Underfitting) 상태입니다! alpha 값을 줄여주세요.")
    else:
        st.success("✅ 훈련 세트와 테스트 세트의 성능 격차가 작아 일반화 성능이 매우 조율된 안정적인 상태입니다!")

# R^2 점수 vs alpha 그래프 시각화 (x축 로그 스케일 적용)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(alpha_list, train_scores, label="Train R² Score", color="#2F80ED", marker="o", linewidth=2, alpha=0.8)
ax.plot(alpha_list, test_scores, label="Test R² Score", color="#F2994A", marker="s", linewidth=2, alpha=0.8)

# 현재 선택된 alpha 지점 표시
ax.scatter([alpha_val], [selected_train_score], color="#EB5757", edgecolors="k", s=150, zorder=5, label=f"현재 선택된 훈련 지점 (alpha={alpha_val})")
ax.scatter([alpha_val], [selected_test_score], color="#27AE60", edgecolors="k", s=150, zorder=5, label=f"현재 선택된 테스트 지점")

ax.set_xscale("log")
ax.set_title(f"{model_type.split()[0]} 규제 강도(alpha)에 따른 R² Score 결정계수 곡선", fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("규제 강도 (Alpha, Log Scale)", fontsize=10)
ax.set_ylabel("결정 계수 (R² Score)", fontsize=10)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
st.pyplot(fig)

st.divider()

# 3. 특성 가중치 분석
st.write("### 3. 특성 가중치(Regression Coefficients) 분석")
st.write("선형 회귀 방정식에서 도출된 각 피처의 가중치(`coef_`)를 확인합니다. 규제 강도 $\\alpha$가 커질수록 회귀 계수의 크기가 억제(Ridge)되거나 일부 불필요한 계수가 완전히 0으로 수축(Lasso)합니다.")

# 가중치 추출
fitted_model = best_pipeline.named_steps[model_type.split()[0].lower()]
coefs = fitted_model.coef_

# DataFrame 생성 및 절댓값 기준 내림차순 정렬
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "한글 변수 설명": [
        "소득 중간값 (MedInc)",
        "주택 연식 중간값 (HouseAge)",
        "평균 방 개수 (AveRooms)",
        "평균 침실 개수 (AveBedrms)",
        "총 인구수 (Population)",
        "평균 구성원 수 (AveOccup)",
        "위도 (Latitude)",
        "경도 (Longitude)"
    ],
    "Coefficient (회귀 계수)": coefs,
    "Absolute Coefficient (절댓값)": np.abs(coefs)
})
coef_df_sorted = coef_df.sort_values(by="Absolute Coefficient (절댓값)", ascending=False).reset_index(drop=True)

col_table, col_chart = st.columns([4, 5])

with col_table:
    st.write("#### 📊 가중치 내림차순 정렬 데이터")
    st.dataframe(
        coef_df_sorted.style.format({
            "Coefficient (회귀 계수)": "{:,.5f}",
            "Absolute Coefficient (절댓값)": "{:,.5f}"
        }),
        use_container_width=True
    )

    # Lasso의 희소성 정보 표시
    if "Lasso" in model_type:
        zero_count = np.sum(coefs == 0)
        if zero_count > 0:
            st.info(f"💡 **Lasso 규제 효과:** 8개 피처 중 총 **{zero_count}개**의 가중치가 정확하게 **0.0**으로 소멸하여 자동으로 특성 선택이 수행되었습니다.")
        else:
            st.info("💡 **Lasso 규제 효과:** 현재 설정된 alpha 값이 작아서 가중치가 완전히 0으로 소멸한 특성이 없습니다. alpha를 더 올려보세요.")

with col_chart:
    st.write("#### 📊 가중치 영향도 방향 및 강도 시각화")

    # 변수 영향력 그래프 작성을 위해 가중치 값 순으로 정렬
    coef_df_for_plot = coef_df.sort_values(by="Coefficient (회귀 계수)")
    colors = ["#EB5757" if c < 0 else "#2F80ED" for c in coef_df_for_plot["Coefficient (회귀 계수)"]]

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5))
    bars = ax_bar.barh(coef_df_for_plot["한글 변수 설명"], coef_df_for_plot["Coefficient (회귀 계수)"], color=colors, edgecolor="k", alpha=0.85)

    # 0선 추가
    ax_bar.axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7)

    # 가로 막대 우측/좌측에 텍스트 값 표시
    for bar in bars:
        width = bar.get_width()
        if width >= 0:
            ax_bar.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"+{width:.3f}",
                        va='center', ha='left', fontsize=9, fontweight='bold', color='#2F80ED')
        else:
            ax_bar.text(width - 0.02, bar.get_y() + bar.get_height()/2, f"{width:.3f}",
                        va='center', ha='right', fontsize=9, fontweight='bold', color='#EB5757')

    ax_bar.set_title(f"{model_type.split()[0]} 가중치 영향력 비교분석 (alpha={alpha_val})", fontsize=11, fontweight="bold", pad=12)
    ax_bar.set_xlabel("회귀 계수 (Coefficient)", fontsize=10)
    ax_bar.grid(True, axis="x", linestyle="--", alpha=0.5)

    # 텍스트 여백을 위해 가로 범위 자동 조율
    max_val = np.max(np.abs(coefs))
    ax_bar.set_xlim(-max_val - 0.25, max_val + 0.25)

    plt.tight_layout()
    st.pyplot(fig_bar)

st.divider()

# 4. 실시간 주택 가격 예측 시뮬레이션
st.write("### 4. 캘리포니아 주택 가격 실시간 예측 시뮬레이션")
st.write("조사 구역의 속성 정보를 하단 슬라이더로 조절하면, 위에서 학습된 `StandardScaler + 규제 모델` 파이프라인을 통과시켜 주택 가격 중간값($MedHouseVal$)의 예측치를 실시간 산출합니다.")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    med_inc = st.slider("🔑 가구 소득 중간값 (MedInc) *($10,000 단위)*", min_value=0.5, max_value=15.0, value=3.87, step=0.1)
    house_age = st.slider("🏡 주택 연식 중간값 (HouseAge)", min_value=1.0, max_value=52.0, value=29.0, step=1.0)
    ave_rooms = st.slider("🛋️ 가구당 평균 방 개수 (AveRooms)", min_value=1.0, max_value=15.0, value=5.2, step=0.1)

with col_p2:
    ave_bedrms = st.slider("🛏️ 가구당 평균 침실 개수 (AveBedrms)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
    population = st.slider("👥 구역의 총 인구수 (Population)", min_value=5, max_value=10000, value=1166, step=50)
    ave_occup = st.slider("👨‍👩‍👧‍👦 가구당 평균 구성원 수 (AveOccup)", min_value=1.0, max_value=10.0, value=2.8, step=0.1)

with col_p3:
    latitude = st.slider("📍 구역 위도 (Latitude)", min_value=32.5, max_value=42.5, value=34.2, step=0.1)
    longitude = st.slider("📍 구역 경도 (Longitude)", min_value=-124.3, max_value=-114.3, value=-118.5, step=0.1)

# 예측 데이터 변환
input_data = pd.DataFrame([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]], columns=X.columns)
predicted_val = best_pipeline.predict(input_data)[0]

# 실제 주택 가격 예측 단위가 10만 달러($100,000)이므로 달러 척도로 확장 변환
usd_value = predicted_val * 100000
clamped_usd = max(0.0, usd_value) # 선형 근사로 인한 비현실적인 음수 예측 방지

st.write("---")
col_res_lbl, col_res_val = st.columns([1, 1])

with col_res_lbl:
    st.subheader("💡 머신러닝 모델 예측 결과")
    st.write(f"현재 예측에 사용된 모델: **{model_type}** | alpha: **{alpha_val}**")
    st.write("입력하신 조사 구역의 조건 하에, 모델이 예측하는 해당 지역 주택 가격 중간값의 평가 추정치는 우측 수치와 같습니다.")

with col_res_val:
    st.metric(
        label="🏡 예측된 주택 가격 중간값 (Median House Value Estimate)",
        value=f"${clamped_usd:,.2f}",
        delta=f"{predicted_val:.4f} (10만 달러 단위)"
    )
