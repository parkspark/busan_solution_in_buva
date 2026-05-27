import streamlit as st

st.set_page_config(
    page_title="ML_01",
    page_icon="📈",
    layout="wide"
)
TIL_page = st.Page("ML01_Supervise_learning.py", title="정리")
titanic_ML_page = st.Page("ML01_titanic.py", title="적용해보기", icon="🔍")
Quiz_page = st.Page("ML01_quiz.py", title="퀴즈", icon="📝")


ML02_01_scaler = st.Page("ML02_01_scaler.py", title="정리")
ML02_02_knnRegression = st.Page("ML02_02_knnRegression.py", title="농어 무게 예측 - k-NN 및 선형 회귀", icon="📈")
ML02_03_PolynomialRegression = st.Page("ML02_03_Polynomial Regression.py", title="다항 회귀와 규제", icon="⚡")
ML02_04_Califonia = st.Page("ML02_04_Califonia.py", title="캘리포니아 집값 예측 - 다중 회귀 및 규제", icon="🏡")
ML02_05_Quiz = st.Page("ML02_05_Quiz.py", title="퀴즈", icon="📝")
ML02_06_califonia = st.Page("ML02_06_califonia.py", title="캘리포니아 집값 예측 - Jupyter Notebook 결과", icon="📓")

pg = st.navigation([TIL_page, titanic_ML_page, Quiz_page, 
    ML02_01_scaler, ML02_02_knnRegression, ML02_03_PolynomialRegression, ML02_04_Califonia, ML02_06_califonia, ML02_05_Quiz], position="hidden")

with st.sidebar:
    st.title("📚 머신러닝 학습")
    
    st.subheader("ML 01: 지도학습 기초")
    st.page_link(TIL_page)
    st.page_link(titanic_ML_page)
    st.page_link(Quiz_page)
    
    st.divider() 
    
    st.subheader("ML 02: 회귀 알고리즘")
    st.page_link(ML02_01_scaler)
    st.page_link(ML02_02_knnRegression)
    st.page_link(ML02_03_PolynomialRegression)
    st.page_link(ML02_04_Califonia)
    st.page_link(ML02_06_califonia)
    st.page_link(ML02_05_Quiz)


pg.run()
