import streamlit as st

# 전체 페이지 설정 (각 개별 페이지의 설정이 덮어쓸 수 있지만, 기본 테마로 어두운 테마 사용을 추천합니다)
# st.set_page_config는 각 개별 py 파일 안에 이미 있으므로 여기서는 생략하거나, 네비게이션용으로만 남겨둡니다.

st.sidebar.title("딥러닝 시각화 게임 🎮")
st.sidebar.markdown("직접 클릭하며 배우는 **딥러닝 인터랙티브 시뮬레이터** 모음입니다.")

# DL_game 폴더 내의 py 파일들을 페이지로 등록
pages = {
    "시뮬레이터 목록": [
        st.Page("DL_game/back02.py", title="1. AI 조련사 (순전파/역전파)", icon="🤖"),
        st.Page("DL_game/Vanishing Gradient.py", title="2. 기울기 소멸 시뮬레이터", icon="📉"),
        st.Page("DL_game/Backpropagation.py", title="3. Backprop Hero (초기버전)", icon="🦸"),
    ]
}

# 네비게이션 실행
pg = st.navigation(pages)
pg.run()
