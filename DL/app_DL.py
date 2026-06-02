import streamlit as st

st.set_page_config(
    page_title="Neural Odyssey",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DL 01
DL01_Terms = st.Page("DL001/DL01_terms.py", title="기초 용어 정리", icon="📝")

# Neural Odyssey Games
Stage1 = st.Page("DL001/DL02_perceptron2.py", title="Stage 1: 퍼셉트론의 탄생", icon="🧠")
Stage2 = st.Page("DL001/DL03_logicgate.py", title="Stage 2: 논리의 한계", icon="🧩")
Stage3 = st.Page("DL001/DL04_game.py", title="Stage 3: 숨겨진 층의 힘", icon="✨")
Stage4 = st.Page("DL001/DL05_game.py", title="Stage 4: 신경망 조립술", icon="🛠️")
Stage5 = st.Page("DL001/DL06_game.py", title="Stage 5: 고전 vs 현대", icon="⚖️")
Stage6 = st.Page("DL001/DL07_game.py", title="Stage 6: 이미지의 비밀", icon="👁️")

pg = st.navigation({
    "Deep Learning 기초": [DL01_Terms],
    "Neural Odyssey": [Stage1, Stage2, Stage3, Stage4, Stage5, Stage6]
})
pg.run()
