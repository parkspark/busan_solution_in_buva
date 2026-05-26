import streamlit as st

st.set_page_config(
    page_title="ML_01",
    page_icon="📈",
    layout="wide"
)
TIL_page = st.Page("ML01_Supervise_learning.py", title="정리")
titanic_ML_page = st.Page("ML01_titanic.py", title="적용해보기", icon="🔍")
Quiz_page = st.Page("ML01_quiz.py", title="퀴즈", icon="📝")

pg = st.navigation([TIL_page, titanic_ML_page, Quiz_page])
pg.run()
