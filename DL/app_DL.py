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
Stage7 = st.Page("DL002/DL02_03_game007.py", title="Stage 7: 역전파의 비밀", icon="🔄")
Stage8 = st.Page("DL002/DL02_04_game008.py", title="Stage 8: 기울기 소멸의 위기", icon="📉")
Stage9 = st.Page("DL002/DL02_05_game09.py", title="Stage 9: ReLU의 각성", icon="⚡")
Stage10 = st.Page("DL002/DL02_06_game10.py", title="Stage 10: 깊이의 힘", icon="👁️")
Stage11 = st.Page("DL002/Dl02_07_game11.py", title="Stage 11: 최적의 길 찾기", icon="🧭")

# Interactive Simulators (DL_game)
Sim1 = st.Page("DL_game/back02.py", title="1. AI 조련사 (순전파/역전파)", icon="🤖")
Sim2 = st.Page("DL_game/Vanishing Gradient.py", title="2. 기울기 소멸 시뮬레이터", icon="📉")
Sim3 = st.Page("DL_game/Backpropagation.py", title="3. Backprop Hero", icon="🦸")

# CNN 시각화 도구 (DL004 & DL005)
DL04_01 = st.Page("DL004/DL04_01_LimitationOfDense.py", title="01. Dense Layer의 한계", icon="🧱")
DL04_02 = st.Page("DL004/DL04_02_convolution.py", title="02. 합성곱(Convolution)의 기초", icon="🔍")
DL04_03 = st.Page("DL004/DL04_03_EdgeDetection.py", title="03. 엣지 감지 (Sobel Filter)", icon="🔪")
DL04_04 = st.Page("DL004/DL04_04_Weight Sharing.py", title="04. 가중치 공유 (Weight Sharing)", icon="⚖️")
DL04_05 = st.Page("DL004/DL04_05_Convolution_Layer.py", title="05. Conv2D 파라미터 계산기", icon="🧮")
DL04_06 = st.Page("DL004/DL04_06_padding.py", title="06. 패딩 (Padding: Valid vs Same)", icon="📏")
DL04_07 = st.Page("DL004/DL04_07_stride.py", title="07. 스트라이드 (Stride)", icon="🏃")
DL04_08 = st.Page("DL004/DL04_08_MaxPooling2D.py", title="08. 풀링 (Max/Average Pooling)", icon="🏊")
DL04_09 = st.Page("DL004/DL04_09_CNN_Flow.py", title="09. CNN 파이프라인 흐름도", icon="🌊")
DL04_10 = st.Page("DL004/DL04_10_Spatial Hierarchy.py", title="10. 공간적 계층 구조", icon="🏢")
DL04_11 = st.Page("DL004/DL04_11_3D.py", title="11. 3D 복셀 합성곱 (Voxel)", icon="🧊")
DL04_12 = st.Page("DL004/DL04_12_4D.py", title="12. 4D 시공간 합성곱 (Time+Space)", icon="⏳")

# CNN 심화 및 응용 (DL005)
DL05_01 = st.Page("DL005/DL05_01_ConvFilter.py", title="01. 합성곱 필터 심화", icon="🔍")
DL05_02 = st.Page("DL005/DL05_02_FuctionalAPI.py", title="02. Functional API 모델링", icon="🏗️")
DL05_03 = st.Page("DL005/DL05_03_AI_Xray.py", title="03. AI X-Ray 시각화", icon="🩺")
DL05_04 = st.Page("DL005/DL05_04_DataAugmentation.py", title="04. 데이터 증강 기초", icon="🪄")
DL05_05 = st.Page("DL005/DL05_05_DataAugmentation_practice.py", title="05. 데이터 증강 실습", icon="💻")
DL05_06 = st.Page("DL005/DL05_06_TransferLearning.py", title="06. 전이 학습(Transfer Learning)", icon="♻️")
DL05_07 = st.Page("DL005/DL05_07_YOLO.py", title="07. 객체 탐지(YOLO) 원리 비교", icon="⚡")

pages = {
    "Deep Learning 기초": [DL01_Terms, DL02_Terms],
    "Neural Odyssey Games": [Stage1, Stage2, Stage3, Stage4, Stage5, Stage6, Stage7, Stage8, Stage9, Stage10, Stage11],
    "Interactive Simulators": [Sim1, Sim2, Sim3],
    "CNN 시각화 도구": [DL04_01, DL04_02, DL04_03, DL04_04, DL04_05, DL04_06, DL04_07, DL04_08, DL04_09, DL04_10, DL04_11, DL04_12],
    "CNN 심화 및 응용": [DL05_01, DL05_02, DL05_03, DL05_04, DL05_05, DL05_06, DL05_07]
}

pg = st.navigation(pages)
pg.run()
