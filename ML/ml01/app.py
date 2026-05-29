import streamlit as st

st.set_page_config(
    page_title="ML_01",
    page_icon="📈",
    layout="wide"
)

# ML 00
ML00_Dictionary = st.Page("ML00/ML00_Dictionary.py", title="통합 머신러닝 핵심 개념 사전", icon="📖")

# ML 01
TIL_page = st.Page("ML01/ML01_Supervise_learning.py", title="정리")
titanic_ML_page = st.Page("ML01/ML01_titanic.py", title="적용해보기", icon="🔍")
Quiz_page = st.Page("ML01/ML01_quiz.py", title="퀴즈", icon="📝")

# ML 02
ML02_01_scaler = st.Page("ML02/ML02_01_scaler.py", title="정리")
ML02_02_knnRegression = st.Page("ML02/ML02_02_knnRegression.py", title="농어 무게 예측 - k-NN 및 선형 회귀", icon="📈")
ML02_03_PolynomialRegression = st.Page("ML02/ML02_03_Polynomial Regression.py", title="다항 회귀와 규제", icon="⚡")
ML02_04_Califonia = st.Page("ML02/ML02_04_Califonia.py", title="캘리포니아 집값 예측 - 다중 회귀 및 규제", icon="🏡")
ML02_05_Quiz = st.Page("ML02/ML02_05_Quiz.py", title="퀴즈", icon="📝")
ML02_06_califonia = st.Page("ML02/ML02_06_califonia.py", title="캘리포니아 집값 예측 - Jupyter Notebook 결과", icon="📓")
ML02_07_Terms = st.Page("ML02/ML02_07_Terms.py", title="한 눈에 보는 핵심 개념", icon="🎯")

# ML 03
ML03_01_Terms = st.Page("ML03/ML03_01_Terms.py", title="한 눈에 보는 핵심 개념", icon="🎯")
ML03_02_Fish_LogisticRegression = st.Page("ML03/ML03_02_Fish_LogisticRegression.py", title="7종 생선 분류", icon="🐟")
ML03_03_ConfusionMatrix_titanic = st.Page("ML03/ML03_03_ConfusionMatrix_titanic.py", title="타이타닉 분류 및 평가", icon="🚢")
ML03_04_BrestCancer = st.Page("ML03/ML03_04_BrestCancer.py", title="유방암 진단 임계값 조정", icon="🏥")
ML03_05_jupyter = st.Page("ML03/ML03_05_jupyter.py", title="유방암 진단 - Jupyter Notebook 결과", icon="📓")
ML03_06_Quiz = st.Page("ML03/ML03_06_Quiz.py", title="퀴즈", icon="📝")

# Navigation setup
pg = st.navigation(
    [
        ML00_Dictionary, 
        TIL_page, 
        titanic_ML_page, 
        Quiz_page, 
        ML02_01_scaler, 
        ML02_02_knnRegression, 
        ML02_03_PolynomialRegression, 
        ML02_04_Califonia, 
        ML02_06_califonia, 
        ML02_07_Terms, 
        ML02_05_Quiz, 
        ML03_01_Terms, 
        ML03_02_Fish_LogisticRegression, 
        ML03_03_ConfusionMatrix_titanic, 
        ML03_04_BrestCancer, 
        ML03_05_jupyter, 
        ML03_06_Quiz
    ], 
    position="hidden"
)

# Sidebar UI
with st.sidebar:
    st.title("📚 머신러닝 학습")
    
    st.subheader("📖 전체 용어 사전")
    st.page_link(ML00_Dictionary)
    
    st.divider()
    
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
    st.page_link(ML02_07_Terms)
    st.page_link(ML02_05_Quiz)

    st.divider() 
    
    st.subheader("ML 03: 분류 알고리즘")
    st.page_link(ML03_01_Terms)
    st.page_link(ML03_02_Fish_LogisticRegression)
    st.page_link(ML03_03_ConfusionMatrix_titanic)
    st.page_link(ML03_04_BrestCancer)
    st.page_link(ML03_05_jupyter)
    st.page_link(ML03_06_Quiz)

pg.run()
