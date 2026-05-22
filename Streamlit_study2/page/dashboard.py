import streamlit as st
st.title("📊 대시보드")
st.write("여기에 차트가 들어갑니다.")

st.header(f"{st.session_state.get('user_name')}님의 대시보드")