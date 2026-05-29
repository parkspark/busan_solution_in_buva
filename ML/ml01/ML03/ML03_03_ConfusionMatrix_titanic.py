import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

st.title("🚢 타이타닉 불균형 분류 — Confusion Matrix + 평가지표 보고서")
st.markdown("정확도의 함정을 직접 체험하고 `classification_report`와 `ROC-AUC`로 모델을 제대로 평가해 봅니다.")

# 1. 데이터 로드 및 전처리
st.subheader("1. 데이터 전처리 (`dropna` → 714개 확인)")
titanic = sns.load_dataset('titanic')

# 주요 컬럼 추출 후 결측치 제거
titanic_clean = titanic[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']].dropna()
st.write(f"- 원본 데이터 크기: **{len(titanic)}개**")
st.write(f"- 결측치 제거 후 크기: **{len(titanic_clean)}개**")

# 범주형 데이터 변환 (One-Hot Encoding)
titanic_clean = pd.get_dummies(titanic_clean, columns=['sex'], drop_first=True)

X = titanic_clean.drop('survived', axis=1)
y = titanic_clean['survived']

# 2. 데이터 분할 (stratify 적용)
st.subheader("2. 데이터 분할 (`stratify=y` 옵션)")
# stratify=y 옵션으로 불균형 비율 유지
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

col1, col2 = st.columns(2)
with col1:
    st.write("Train 타겟(생존/사망) 비율:")
    st.dataframe(y_train.value_counts(normalize=True).round(3))
with col2:
    st.write("Test 타겟(생존/사망) 비율:")
    st.dataframe(y_test.value_counts(normalize=True).round(3))

st.info("💡 **포인트:** `stratify=y`를 설정했기 때문에 훈련 세트와 테스트 세트의 사망(0)과 생존(1) 비율이 동일하게 유지되었습니다.")

# 3. 모델 학습
st.subheader("3. 로지스틱 회귀 모델 학습")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

train_acc = lr.score(X_train, y_train)
test_acc = lr.score(X_test, y_test)
st.write(f"- **Train Accuracy:** {train_acc:.4f}")
st.write(f"- **Test Accuracy:** {test_acc:.4f}")

# 4. 무조건 사망 예측의 정확도 직접 계산 (정확도의 함정)
st.subheader("4. 🚨 정확도의 함정: '무조건 사망' 예측 테스트")
# 테스트 세트에서 무조건 사망(0)으로 예측했을 때의 정확도 계산
all_dead_pred = np.zeros_like(y_test)
all_dead_acc = (all_dead_pred == y_test).mean()

st.warning(f"만약 모델이 테스트 세트의 모든 승객을 **'무조건 사망(0)'** 했다고 예측한다면? \n\n"
           f"모델 학습을 전혀 하지 않아도 정확도가 **{all_dead_acc:.4f} ({all_dead_acc*100:.1f}%)** 나옵니다.")
st.write("→ 이처럼 불균형 데이터(사망자가 생존자보다 훨씬 많은 데이터)에서는 '정확도(Accuracy)'만 믿으면 모델의 실제 성능(생존자를 얼마나 잘 찾는지)을 착각할 위험이 있습니다.")

# 5. Confusion Matrix & Classification Report
st.subheader("5. Confusion Matrix & Classification Report")
y_pred = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['실제 사망(0)', '실제 생존(1)'], columns=['예측 사망(0)', '예측 생존(1)'])
st.write("✅ **Confusion Matrix (오차 행렬 직접 읽기):**")
st.dataframe(cm_df)
st.markdown("""
- **TN (True Negative):** 실제 사망자를 사망으로 정확히 맞힌 수
- **TP (True Positive):** 실제 생존자를 생존으로 정확히 맞힌 수
- **FP (False Positive):** 실제 사망자인데 생존했다고 틀리게 예측한 수 (과잉 경보)
- **FN (False Negative):** 실제 생존자인데 사망했다고 틀리게 예측한 수 (놓침)
""")

st.write("✅ **Classification Report (`target_names` 설정):**")
# target_names 설정하여 한글로 출력
report = classification_report(y_test, y_pred, target_names=['사망', '생존'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# 6. ROC-AUC
st.subheader("6. ROC-AUC 스코어")
# AUC 계산 시 predict_proba[:, 1] (생존 확률) 사용
y_prob = lr.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)

st.success(f"**ROC-AUC Score:** {auc_score:.4f}")
st.write("→ 클래스 불균형에 흔들리지 않고, 임계값 변화에 따른 모델의 전반적인 분류 능력을 0.5 ~ 1.0 사이로 나타내는 종합 평가 지표입니다.")
