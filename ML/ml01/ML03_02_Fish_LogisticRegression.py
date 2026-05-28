import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os

st.title("🐟 7종 생선 LogisticRegression — 확률 예측 보고서")
st.markdown("LogisticRegression으로 7종 생선 다중 분류를 학습하고 `predict_proba` 결과를 해석한다.")

# 데이터 로드
st.subheader("1. 데이터 로드 및 확인")
file_path = '../ml02/Fish.csv'
if not os.path.exists(file_path):
    # 폴더 구조에 따라 절대 경로 사용
    file_path = r'c:\Users\금정산2-PC15\Desktop\busan_solution_in_buva\ML\ml02\Fish.csv'

fish = pd.read_csv(file_path)

st.write("`fish['Species'].value_counts()` 로 종별 샘플 수 확인:")
st.dataframe(fish['Species'].value_counts())

# 데이터 전처리
st.subheader("2. 데이터 전처리 (StandardScaler)")
X = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']]
y = fish['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

scaler = StandardScaler()
# StandardScaler: train에만 fit_transform, test는 transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.info("""💡 **StandardScaler 주의점:** `train` 세트에만 `fit_transform`을 적용하고, \n
`test` 세트에는 `transform`만 적용해야 데이터 누수(Data Leakage)를 막을 수 있습니다.""")

# 모델 학습
st.subheader("3. 로지스틱 회귀 모델 학습 및 클래스 확인")
# 다중 분류의 경우 최적화를 위해 규제 매개변수 C와 max_iter를 조절하기도 함
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(X_train_scaled, y_train)

st.write("`lr.classes_` 출력 후 열 순서 확인:")
st.code(lr.classes_)
st.write("✅ 위에서 확인한 열 순서는 **알파벳 순서**로 자동 정렬된 상태입니다. `predict_proba`의 열도 이 순서를 따릅니다.")

# 확률 예측 결과
st.subheader("4. 확률 예측 결과 (`predict_proba`) DataFrame 변환")
st.markdown("테스트 세트의 처음 5개 샘플에 대한 예측 확률입니다.")

proba = lr.predict_proba(X_test_scaled[:5])
# 결과를 DataFrame으로 변환
proba_df = pd.DataFrame(np.round(proba, decimals=3), columns=lr.classes_)

st.dataframe(proba_df)
st.write("✅ Softmax 함수를 통과했으므로, 각 행의 확률을 모두 더하면 **1.0 (100%)**이 됩니다.")

# 정확도 평가
st.subheader("5. train / test 정확도 비교")
train_score = lr.score(X_train_scaled, y_train)
test_score = lr.score(X_test_scaled, y_test)

col1, col2 = st.columns(2)
col1.metric("Train Accuracy", f"{train_score:.4f}")
col2.metric("Test Accuracy", f"{test_score:.4f}")

st.write("→ 훈련 세트와 테스트 세트의 정확도 차이를 통해 모델이 적절히 피팅되었는지(High Bias / High Variance가 없는지) 진단할 수 있습니다.")
