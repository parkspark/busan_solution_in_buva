import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

st.title("🏥 유방암 데이터셋 — 의료 분류 + FN 최소화 전략")
st.markdown("`load_breast_cancer`의 label 방향을 이해하고, 생명과 직결된 FN(악성 놓침) 최소화 관점에서 임계값을 조정해 봅니다.")

# 1. 데이터 로드 및 레이블 확인
st.subheader("1. 데이터 로드 및 레이블(Target) 방향 확인")
data = load_breast_cancer()

st.write("`data.target_names` 출력 결과:")
st.code(data.target_names)
st.write("👉 사이킷런의 유방암 데이터는 특이하게도 **`0 = malignant(악성)`, `1 = benign(양성)`**으로 매핑되어 있습니다.")

st.info("💡 **주의:** 보통 질병 유무를 판별할 때는 '질병 있음(악성)'을 1(Positive)로 두는 것이 직관적이고 평가지표(Recall 등) 해석에 유리합니다. 따라서 모델 학습 전에 타겟 레이블을 뒤집어 줍니다.")

# 레이블 뒤집기: 1=악성, 0=양성
st.code("y = 1 - data.target")
y = 1 - data.target
X = data.data

st.write("적용 후 타겟 분포:")
st.write(f"- 양성(0, Benign): {np.sum(y == 0)}개")
st.write(f"- 악성(1, Malignant): {np.sum(y == 1)}개")

# 2. 데이터 분할
st.subheader("2. 데이터 분할 (`stratify=y` 적용)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
st.write("악성과 양성 데이터의 비율이 다르므로 `stratify=y`를 사용하여 Train/Test 분할 시 클래스 비율을 동일하게 맞췄습니다.")

# 3. 모델 학습
st.subheader("3. 로지스틱 회귀 학습 (기본 임계값 0.5)")
lr = LogisticRegression(random_state=42, max_iter=3000)
lr.fit(X_train, y_train)

# 기본 임계값 0.5 예측
y_prob = lr.predict_proba(X_test)[:, 1]
y_pred_default = lr.predict(X_test)

col1, col2 = st.columns(2)
with col1:
    st.write("**기본 임계값(0.5) 오차 행렬**")
    cm_default = confusion_matrix(y_test, y_pred_default)
    st.dataframe(pd.DataFrame(cm_default, index=['실제 양성(0)', '실제 악성(1)'], columns=['예측 양성(0)', '예측 악성(1)']))
    fn_default = cm_default[1, 0]
    st.write(f"🚨 **놓친 악성 환자 (FN):** {fn_default}명")

with col2:
    st.write("**기본 임계값(0.5) 분류 보고서**")
    report_default = classification_report(y_test, y_pred_default, target_names=['양성(0)', '악성(1)'], output_dict=True)
    st.dataframe(pd.DataFrame(report_default).transpose())

# 4. 임계값 조정 (FN 최소화 전략)
st.subheader("4. 임계값 조정: FN(악성 놓침) 최소화 전략")
st.markdown("의료 도메인에서는 질병을 놓치는 것(FN)이 치명적입니다. 따라서 악성이라고 판정하는 **커트라인(임계값)을 낮춰서** 덜 확실해도 일단 악성(1)으로 예측하게 만듭니다.")

# 임계값 0.3 적용
st.code("y_pred_custom = (y_prob >= 0.3).astype(int)")
y_pred_custom = (y_prob >= 0.3).astype(int)

col3, col4 = st.columns(2)
with col3:
    st.write("**조정된 임계값(0.3) 오차 행렬**")
    cm_custom = confusion_matrix(y_test, y_pred_custom)
    st.dataframe(pd.DataFrame(cm_custom, index=['실제 양성(0)', '실제 악성(1)'], columns=['예측 양성(0)', '예측 악성(1)']))
    fn_custom = cm_custom[1, 0]
    st.write(f"✅ **놓친 악성 환자 (FN):** {fn_custom}명 (감소!)")

with col4:
    st.write("**조정된 임계값(0.3) 분류 보고서**")
    report_custom = classification_report(y_test, y_pred_custom, target_names=['양성(0)', '악성(1)'], output_dict=True)
    st.dataframe(pd.DataFrame(report_custom).transpose())

# 5. Recall 변화 이유 설명
st.subheader("💡 임계값 변경 후 Recall(재현율)은 왜 상승했을까?")
st.success("""
**임계값을 0.5에서 0.3으로 낮추면, 모델이 악성(1)이라고 판정하기 위한 커트라인이 매우 관대해집니다.**  
따라서 모델이 더 많은 환자를 악성으로 진단(Positive 예측 증가)하게 되고, 이로 인해 실제 악성 환자 중에서 놓치는 비율(FN)이 줄어들어 **결과적으로 전체 악성 환자를 찾아내는 비율인 Recall(재현율)은 필연적으로 상승**하게 됩니다. (대신 정상 환자를 악성으로 잘못 진단하는 과잉 경보(FP)가 늘어나 Precision은 하락하는 트레이드오프가 발생합니다.)
""")
