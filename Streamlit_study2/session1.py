# uv run streamlit run session1.py



import streamlit as st
import pandas as pd
import plotly.express as px

#1
count_zero = 0
if st.button("클릭"):
    count_zero += 1
st.write("클릭 횟수 : ", count_zero)
# 1까지만 가고 멈추는 이유. 
# rerun 되기때문에 count=0 으로 계속 실행된다.
# 이를 해결하려면 session_state를 사용해야 한다.

#2
st.subheader("session_state 사용")
if 'count' not in st.session_state:
    st.session_state['count'] = 0
# Session_state에서 'count'가 없으면 0으로 초기화 하라는 뜻.     
# 즉, session_state에 'count'가 있으면 0으로 초기화하지 말라는 뜻.
# (왜냐하면 session_state는 rerun이 되어도 유지되기 때문이다.)
# 읽기
st.write("클릭 수:", st.session_state['count'])
# 쓰기
if st.button("클릭2"):
    st.session_state['count'] += 1
if st.button("초기화"):
    st.session_state['count'] = 0

st.divider()


# 1-4 실용패턴 A - 좋아요 버튼
if 'liked' not in st.session_state:
    st.session_state['liked'] = False

label = "❤️ 좋아요 취소" if st.session_state['liked'] else "🤍 좋아요"
if st.button(label):
    st.session_state['liked'] = not st.session_state['liked']

if st.session_state['liked']:
    st.success("좋아요를 눌렀습니다!")

# 2 부터는 첫번쨰 클릭을 해도 1로 바로 늘어나지않고 한번의 액션 늦게 1카운트된다.
# 이후 초기화 버튼을 눌러도 1카운팅이 된뒤에 한번 더 누르면 초기화된다. => 해결방법은 뭘까?
'''
Streamlit에서 상태 변경(세션 스테이트 업데이트)이 한 박자 늦게 반영되는 현상은 Streamlit의 렌더링(Rerun) 동작 매커니즘 때문에 발생하는 아주 대표적인 현상입니다.

1. 현상이 발생한 이유 (원인)
Streamlit은 사용자가 버튼을 클릭하는 등의 액션을 취할 때마다, 파이썬 스크립트를 **위에서 아래로(Top-to-Bottom) 매번 전체 재실행(Rerun)**합니다.

기존 코드의 동작 순서는 다음과 같았습니다:

값 읽기 및 그리기: st.write("클릭 수:", st.session_state['count']) 가 먼저 실행됩니다. (이때 화면에는 이전 상태 값인 0이 그려집니다.)
값 변경(쓰기): 뒤이어서 if st.button("클릭2"): 조건문 안에서 st.session_state['count'] += 1이 실행되어 세션 상태의 값이 1로 올라갑니다.
스크립트가 종료되고 웹 브라우저에 최종 렌더링 결과가 보여집니다.
즉, 화면에 값을 그리는 시점(st.write)이 세션 상태를 실제로 증가시키는 시점(st.button 판정)보다 위에 있었기 때문에, 내부 메모리(Session State)의 값은 올라갔지만 화면에는 바로 보이지 않고 다음 리런 시점에 한 박자 늦게 표시되는 문제가 발생한 것입니다. 초기화 버튼도 마찬가지로 화면을 먼저 그린 뒤에 0으로 초기화되었기 때문에 한 박자 늦게 동작했습니다.

2. 가장 확실한 해결 방법: 콜백(Callback) 함수 사용
이 문제를 완벽하게 해결하는 Streamlit의 표준 패턴은 콜백(Callback) 함수를 사용하는 것입니다.

st.button의 on_click 옵션에 상태 변경 함수를 등록해 주면, Streamlit은 전체 스크립트를 처음부터 재실행하기 직전에 콜백 함수를 먼저 실행합니다. 덕분에 st.write가 스크립트 위쪽에 있더라도 이미 변경이 완료된 최신 상태 값을 화면에 즉각적으로 그릴 수 있게 됩니다.

# 콜백 함수 정의 (버튼 클릭 시 스크립트 실행 전에 가장 먼저 실행됨)
def increment_count():
    st.session_state['count'] += 1

def reset_count():
    st.session_state['count'] = 0

if 'count' not in st.session_state:
    st.session_state['count'] = 0

# 읽기 (콜백 함수 덕분에 항상 최신 값을 화면에 표시할 수 있습니다)
st.write("클릭 수:", st.session_state['count'])

# 쓰기 (on_click 콜백 적용)
st.button("클릭2", on_click=increment_count)
st.button("초기화", on_click=reset_count)


if 'liked' not in st.session_state:
    st.session_state['liked'] = False

def toggle_liked():
    st.session_state['liked'] = not st.session_state['liked']

# 콜백 함수를 지정하면 클릭 즉시 세션 상태가 변경되므로 
# 버튼 라벨과 안내 메시지가 한 박자 늦지 않고 즉시 반영됩니다.
label = "❤️ 좋아요 취소" if st.session_state['liked'] else "🤍 좋아요"
st.button(label, on_click=toggle_liked)

if st.session_state['liked']:
    st.success("좋아요를 눌렀습니다!")
    
'''