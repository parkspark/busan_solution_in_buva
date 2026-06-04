import streamlit as st

st.set_page_config(
    page_title="Neural Odyssey",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DL 01
DL01_Terms = st.Page("DL001/DL01_terms.py", title="Day 1: 기초 용어 정리", icon="📝")

# DL 02
DL02_Terms = st.Page("DL002/DL02_01_terms.py", title="Day 2: 핵심 개념 상세", icon="📚")

# Neural Odyssey Games
Stage1 = st.Page("DL001/DL02_perceptron2.py", title="Stage 1: 퍼셉트론의 탄생", icon="🧠")
Stage2 = st.Page("DL001/DL03_logicgate.py", title="Stage 2: 논리의 한계", icon="🧩")
Stage3 = st.Page("DL001/DL04_game.py", title="Stage 3: 숨겨진 층의 힘", icon="✨")
Stage4 = st.Page("DL001/DL05_game.py", title="Stage 4: 신경망 조립술", icon="🛠️")
Stage5 = st.Page("DL001/DL06_game.py", title="Stage 5: 고전 vs 현대", icon="⚖️")
Stage6 = st.Page("DL001/DL07_game.py", title="Stage 6: 이미지의 비밀", icon="👁️")

# Neural Odyssey Games (Day 2)
Stage7 = st.Page("DL002/dL02_03_game007.py", title="Stage 7: 역전파의 비밀", icon="🔄")
Stage8 = st.Page("DL002/DL02_04_game008.py", title="Stage 8: 기울기 소멸의 위기", icon="📉")
Stage9 = st.Page("DL002/DL02_05_game09.py", title="Stage 9: ReLU의 각성", icon="⚡")
Stage10 = st.Page("DL002/DL02_06_game10.py", title="Stage 10: 깊이의 힘", icon="👁️")
Stage11 = st.Page("DL002/Dl02_07_game11.py", title="Stage 11: 최적의 길 찾기", icon="🧭")

# Interactive Simulators (DL_game)
Sim1 = st.Page("DL_game/back02.py", title="1. AI 조련사 (순전파/역전파)", icon="🤖")
Sim2 = st.Page("DL_game/Vanishing Gradient.py", title="2. 기울기 소멸 시뮬레이터", icon="📉")
Sim3 = st.Page("DL_game/Backpropagation.py", title="3. Backprop Hero", icon="🦸")

pg = st.navigation({
    "Deep Learning 기초": [DL01_Terms, DL02_Terms],
    "Neural Odyssey": [Stage1, Stage2, Stage3, Stage4, Stage5, Stage6, Stage7, Stage8, Stage9, Stage10, Stage11],
    "인터랙티브 시뮬레이터": [Sim1, Sim2, Sim3]
})
pg.run()
