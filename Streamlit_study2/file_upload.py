import streamlit as st
import pandas as pd

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일 입력하세요", ["CSV"])

# 파일이 업로드 되었을 때
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())


# 파일 업로드
uploaded_file = st.file_uploader("사진 파일 입력하세요", ["jfif", "jpg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file)