# uv run streamlit run session2.py

import streamlit as st

age_range = st.slider('나이 범위', 0, 80, (0,80),
key='age_range')
# 위젯의 매개변수를 key로 지정할 경우 session_state의 key로 자종 저장 한다.
# 초기화 가드 대신 최초 실행시 value 값을 초기값으로 사용한다.
# 데이터 변경시 -> rerun -> 마지막 session state 값으로 불러온다.


st.write('선택하신 나이대는: ', st.session_state['age_range'])
