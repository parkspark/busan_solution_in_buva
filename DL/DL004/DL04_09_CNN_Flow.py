import streamlit as st
import pandas as pd
import time

# ==============================================================================
# SESSION STATE INITIALIZATION & LANGUAGE
# ==============================================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'KOR'

t = {
    'ENG': {
        'page_title': "CNN Architecture Visualizer",
        'title': "CNN Architecture Pipeline Flow",
        'subtitle': "An interactive visualizer mapping a 2D image input to a final classification prediction, accompanied by the corresponding Keras code.",
        'btn_step': "▶ Step-by-Step Execution",
        'btn_reset': "Reset Pipeline",
        'btn_summary': "Show model.summary()",
        'btn_hide_summary': "Hide model.summary()",
        'code_title': "### Keras / TensorFlow Implementation",
        'summary_title': "### Simulated `model.summary()`",
        'layer_details': "### Active Layer Details",
        'in_shape': "Input Shape",
        'out_shape': "Output Shape",
        'params': "Parameter Count"
    },
    'KOR': {
        'page_title': "CNN 아키텍처 시각화 도구",
        'title': "CNN 아키텍처 파이프라인 흐름도",
        'subtitle': "2D 이미지 입력부터 최종 분류 예측까지 데이터가 흐르는 과정을 시각화",
        'btn_step': "▶ 파이프라인 실행 애니메이션",
        'btn_reset': "흐름 초기화",
        'btn_summary': "model.summary() 보기",
        'btn_hide_summary': "model.summary() 숨기기",
        'code_title': "### Keras / TensorFlow 구현 코드",
        'summary_title': "### 가상 `model.summary()` 출력",
        'layer_details': "### 현재 활성화된 계층 세부 정보",
        'in_shape': "입력 형태 (Input Shape)",
        'out_shape': "출력 형태 (Output Shape)",
        'params': "파라미터 수 (Params)"
    }
}

st.set_page_config(page_title=t[st.session_state.lang]['page_title'], layout="wide")

# ==============================================================================
# HEADER BANNER & LANGUAGE TOGGLE
# ==============================================================================
header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.title(t[st.session_state.lang]['title'])
    st.markdown(t[st.session_state.lang]['subtitle'])

with header_col2:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    lang_col2, lang_col1 = st.columns(2)
    with lang_col1:
        if st.button("ENG", type="primary" if st.session_state.lang == 'ENG' else "secondary", use_container_width=True, key="lang_eng"):
            st.session_state.lang = 'ENG'
            st.rerun()
    with lang_col2:
        if st.button("KOR", type="primary" if st.session_state.lang == 'KOR' else "secondary", use_container_width=True, key="lang_kor"):
            st.session_state.lang = 'KOR'
            st.rerun()

st.divider()

# ==============================================================================
# DATA DEFINITIONS
# ==============================================================================
layers = [
    {"name": "Input", "type": "input", "in_shape": "(32, 32, 3)", "out_shape": "(32, 32, 3)", "params": 0, "desc": "Receives raw RGB image data."},
    {"name": "Conv2D (32, 3x3)", "type": "conv", "in_shape": "(32, 32, 3)", "out_shape": "(30, 30, 32)", "params": 896, "desc": "Extracts spatial features using 32 filters."},
    {"name": "MaxPooling2D (2x2)", "type": "pool", "in_shape": "(30, 30, 32)", "out_shape": "(15, 15, 32)", "params": 0, "desc": "Downsamples spatial dimensions to reduce computation."},
    {"name": "Flatten", "type": "flat", "in_shape": "(15, 15, 32)", "out_shape": "(7200,)", "params": 0, "desc": "Converts 2D feature maps into a 1D vector."},
    {"name": "Dense (64)", "type": "dense", "in_shape": "(7200,)", "out_shape": "(64,)", "params": 460864, "desc": "Fully connected layer for high-level reasoning."},
    {"name": "Dense (10, Softmax)", "type": "dense", "in_shape": "(64,)", "out_shape": "(10,)", "params": 650, "desc": "Final classification into 10 probability classes."}
]

keras_code = """import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.InputLayer(input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
"""

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================
if 'cnn_step' not in st.session_state:
    st.session_state.cnn_step = 0
if 'cnn_autoplay' not in st.session_state:
    st.session_state.cnn_autoplay = False
if 'show_summary' not in st.session_state:
    st.session_state.show_summary = False

# ==============================================================================
# CONTROLS
# ==============================================================================
ctrl1, ctrl2, ctrl3, _ = st.columns([2, 2, 2, 4])

with ctrl1:
    if st.button(t[st.session_state.lang]['btn_step'], disabled=st.session_state.cnn_autoplay, use_container_width=True):
        if st.session_state.cnn_step >= len(layers) - 1:
            st.session_state.cnn_step = 0
        st.session_state.cnn_autoplay = True
        st.rerun()

with ctrl2:
    if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
        st.session_state.cnn_step = 0
        st.session_state.cnn_autoplay = False
        st.rerun()

with ctrl3:
    summary_btn_label = t[st.session_state.lang]['btn_hide_summary'] if st.session_state.show_summary else t[st.session_state.lang]['btn_summary']
    if st.button(summary_btn_label, use_container_width=True):
        st.session_state.show_summary = not st.session_state.show_summary
        st.rerun()

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# PIPELINE VISUALIZATION
# ==============================================================================
def render_pipeline(active_idx):
    # Semantic colors using generic rgba
    colors = {
        "input": "rgba(255, 153, 0, 0.2)",    # Orange
        "conv": "rgba(54, 162, 235, 0.2)",     # Blue
        "pool": "rgba(75, 192, 192, 0.2)",     # Teal
        "flat": "rgba(153, 102, 255, 0.2)",    # Purple
        "dense": "rgba(255, 99, 132, 0.2)"     # Red
    }
    
    html = "<div style='display: flex; flex-direction: row; align-items: center; justify-content: flex-start; overflow-x: auto; padding: 25px 10px;'>"
    
    for i, layer in enumerate(layers):
        bg = colors[layer['type']]
        border = "3px solid currentColor" if i == active_idx else "1px solid currentColor"
        weight = "bold" if i == active_idx else "normal"
        scale = "transform: scale(1.15);" if i == active_idx else "transform: scale(1.0);"
        opacity = "1.0" if i <= active_idx else "0.4"
        
        box = f"<div style='background-color: {bg}; border: {border}; font-weight: {weight}; {scale} opacity: {opacity}; padding: 15px 15px; margin: 0 5px; border-radius: 8px; text-align: center; transition: all 0.3s ease; white-space: nowrap; font-size: 1.1rem;'>{layer['name']}</div>"
        html += box
        
        if i < len(layers) - 1:
            arrow_opacity = "1.0" if i < active_idx else "0.3"
            arrow = f"<div style='font-size: 24px; margin: 0 10px; opacity: {arrow_opacity}; transition: all 0.3s ease;'>➔</div>"
            html += arrow
            
    html += "</div>"
    return html

st.markdown(render_pipeline(st.session_state.cnn_step), unsafe_allow_html=True)
st.divider()

# ==============================================================================
# DETAILS & CODE VIEW
# ==============================================================================
col_details, col_code = st.columns([1, 1])

with col_details:
    st.markdown(t[st.session_state.lang]['layer_details'])
    active_layer = layers[st.session_state.cnn_step]
    
    # Render detail cards
    st.info(f"**{active_layer['name']}**\n\n{active_layer['desc']}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric(t[st.session_state.lang]['in_shape'], active_layer['in_shape'])
    m2.metric(t[st.session_state.lang]['out_shape'], active_layer['out_shape'])
    m3.metric(t[st.session_state.lang]['params'], f"{active_layer['params']:,}")
    
with col_code:
    st.markdown(t[st.session_state.lang]['code_title'])
    st.code(keras_code, language='python')

# ==============================================================================
# MODEL SUMMARY
# ==============================================================================
if st.session_state.show_summary:
    st.divider()
    st.markdown(t[st.session_state.lang]['summary_title'])
    
    summary_data = [
        {"Layer (type)": "conv2d (Conv2D)", "Output Shape": "(None, 30, 30, 32)", "Param #": 896},
        {"Layer (type)": "max_pooling2d (MaxPooling2D)", "Output Shape": "(None, 15, 15, 32)", "Param #": 0},
        {"Layer (type)": "flatten (Flatten)", "Output Shape": "(None, 7200)", "Param #": 0},
        {"Layer (type)": "dense (Dense)", "Output Shape": "(None, 64)", "Param #": 460864},
        {"Layer (type)": "dense_1 (Dense)", "Output Shape": "(None, 10)", "Param #": 650}
    ]
    df = pd.DataFrame(summary_data)
    
    st.table(df)
    
    total_params = 896 + 460864 + 650
    st.markdown(f"**Total params:** {total_params:,}")
    st.markdown(f"**Trainable params:** {total_params:,}")
    st.markdown("**Non-trainable params:** 0")

# ==============================================================================
# AUTO-PLAY LOGIC
# ==============================================================================
if st.session_state.cnn_autoplay and st.session_state.cnn_step < len(layers) - 1:
    time.sleep(1.0)
    st.session_state.cnn_step += 1
    st.rerun()
elif st.session_state.cnn_autoplay and st.session_state.cnn_step >= len(layers) - 1:
    st.session_state.cnn_autoplay = False
    st.rerun()
