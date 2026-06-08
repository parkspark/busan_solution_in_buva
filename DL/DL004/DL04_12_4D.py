import streamlit as st
import numpy as np
import time

# ==============================================================================
# SESSION STATE INITIALIZATION & LANGUAGE
# ==============================================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'KOR'

t = {
    'ENG': {
        'page_title': "4D Spatiotemporal Data Visualizer",
        'title': "4D CNN: Spatiotemporal Convolution",
        'subtitle': "An interactive visualizer demonstrating how a 4D kernel processes 3D volumes changing over time.",
        'slider_t': "Timeline (Temporal Step T)",
        'slider_s': "Spatial Traversal (Step XYZ)",
        'btn_autoplay': "Auto-Play All",
        'diff_map': "Temporal Difference Map ($|V_{t+1} - V_t|$)",
        'vol_t0': "Input Volume at",
        'vol_t1': "Input Volume at",
        'out_vol': "Output Feature Map (at current T)",
        'calc': "4D Element-wise Calculation (Space + Time)",
        'info_title': "Theory: Temporal Correlation & 4D Convolution",
        'info_1': "### 1. Spatiotemporal Data\nData like video or dynamic 3D scans have 4 dimensions: Time, Depth, Height, Width. A 4D kernel slides across all these dimensions simultaneously.",
        'info_2': "### 2. Temporal Correlation\nBy encompassing multiple time steps (e.g., $t$ and $t+1$), the 4D kernel learns how spatial features change over time, capturing motion, velocity, and dynamic structures.",
        'info_3': "### 3. Difference Maps\nDifference maps highlight *what* moved. A 4D convolution natively learns similar temporal gradients, making it powerful for action recognition and motion tracking."
    },
    'KOR': {
        'page_title': "4D 시공간 데이터 시각화 도구",
        'title': "4D CNN: 시공간적 합성곱 (Spatiotemporal Convolution)",
        'subtitle': "시간의 흐름에 따라 변하는 3D 부피를 4D 커널이 어떻게 처리하는지 보여주는 대화형 시각화 도구입니다.",
        'slider_t': "타임라인 (시간 단계, Temporal Step)",
        'slider_s': "공간 탐색 (공간 단계, Spatial Step)",
        'btn_autoplay': "전체 자동 재생",
        'diff_map': "시간적 차이 맵 ($|V_{t+1} - V_t|$)",
        'vol_t0': "입력 부피 (시간",
        'vol_t1': "입력 부피 (시간",
        'out_vol': "출력 특징 맵 (현재 시간 단계)",
        'calc': "4D 요소별 연산 (공간 + 시간 차원 동시 계산)",
        'info_title': "이론: 시간적 상관관계와 4D 합성곱",
        'info_1': "### 1. 시공간 데이터 (Spatiotemporal Data)\n비디오 영상이나 연속적인 3D 의료 스캔은 시간, 깊이, 높이, 너비의 4차원을 가집니다. 4D 커널은 이 모든 차원을 가로지르며 슬라이딩합니다.",
        'info_2': "### 2. 시간적 상관관계 (Temporal Correlation)\n$t$와 $t+1$과 같이 여러 시간 단계를 동시에 포함함으로써, 4D 커널은 공간적 특징이 시간에 따라 어떻게 변하는지(움직임, 속도 등)를 학습할 수 있습니다.",
        'info_3': "### 3. 차이 맵 (Difference Maps)\n차이 맵은 객체가 '어디로 움직였는지'를 시각적으로 강조합니다. 4D 합성곱은 내부적으로 이러한 시간적 변화율(Gradient)을 학습하여 행동 인식 등에 강력한 성능을 발휘합니다."
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

# ==============================================================================
# EDUCATIONAL OVERLAY
# ==============================================================================
with st.expander(t[st.session_state.lang]['info_title']):
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.info(t[st.session_state.lang]['info_1'])
    with col_info2:
        st.warning(t[st.session_state.lang]['info_2'])
    with col_info3:
        st.success(t[st.session_state.lang]['info_3'])

st.divider()

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================
if 'step_t' not in st.session_state:
    st.session_state.step_t = 0
if 'step_s' not in st.session_state:
    st.session_state.step_s = 0
if 'autoplay_4d' not in st.session_state:
    st.session_state.autoplay_4d = False

# ==============================================================================
# CONTROLS
# ==============================================================================
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])

with col_ctrl1:
    step_t = st.slider(t[st.session_state.lang]['slider_t'], 0, 3, st.session_state.step_t)
    if step_t != st.session_state.step_t:
        st.session_state.step_t = step_t
        st.session_state.step_s = 0
        st.rerun()

with col_ctrl2:
    step_s = st.slider(t[st.session_state.lang]['slider_s'], 0, 7, st.session_state.step_s)
    if step_s != st.session_state.step_s:
        st.session_state.step_s = step_s
        st.rerun()

with col_ctrl3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    auto_play = st.toggle(t[st.session_state.lang]['btn_autoplay'], value=st.session_state.autoplay_4d)
    if auto_play != st.session_state.autoplay_4d:
        st.session_state.autoplay_4d = auto_play
        st.rerun()

st.divider()

# ==============================================================================
# 4D DATA GENERATION (5 Time Steps of 3x3x3 Voxels)
# ==============================================================================
np.random.seed(42)
input_vols = np.random.randint(0, 3, size=(5, 3, 3, 3))

# Simulate a moving hot block (value 5) diagonally
for ti in range(5):
    input_vols[ti, min(2, ti//2), min(2, ti//2), min(2, ti//2)] = 5
    if ti < 4:
        input_vols[ti, min(2, ti//2), min(2, ti//2+1), min(2, ti//2)] = 4

# 4D Kernel (2x2x2x2) -> Shape: (Time, Z, Y, X)
kernel = np.array([
    [[[1, 0], [0, -1]], [[0, 1], [-1, 0]]],  # Time 0
    [[[-1, 0], [0, 1]], [[0, -1], [1, 0]]]   # Time 1
])

# Output Volume Shape (4, 2, 2, 2)
out_vols = np.zeros((4, 2, 2, 2), dtype=int)
for ti in range(4):
    for z in range(2):
        for y in range(2):
            for x in range(2):
                region = input_vols[ti:ti+2, z:z+2, y:y+2, x:x+2]
                out_vols[ti, z, y, x] = np.sum(region * kernel)

diff_map = np.abs(input_vols[st.session_state.step_t + 1] - input_vols[st.session_state.step_t])

active_z = st.session_state.step_s // 4
active_y = (st.session_state.step_s % 4) // 2
active_x = st.session_state.step_s % 2

# ==============================================================================
# 3D ISOMETRIC RENDERING FUNCTIONS
# ==============================================================================
def render_4d_slice(volume, az, ay, ax, title, color_rgba, max_val, show_active=True):
    depth = volume.shape[0]
    size = volume.shape[1]
    box_size = 90
    z_spacing = 40
    
    html = f"<h5 style='text-align: center; margin-bottom: 20px; font-weight: bold;'>{title}</h5>"
    html += f"""<div style="perspective: 1200px; width: 100%; display: flex; justify-content: center; padding-top: 20px; padding-bottom: 80px;">
      <div style="transform: rotateX(60deg) rotateZ(-45deg); transform-style: preserve-3d; width: {box_size}px; height: {box_size}px; position: relative;">"""
    
    for z in range(depth):
        translate_z = z * z_spacing
        html += f"<div style='position: absolute; top: 0; left: 0; width: {box_size}px; height: {box_size}px; display: grid; grid-template-columns: repeat({size}, 1fr); grid-template-rows: repeat({size}, 1fr); gap: 2px; transform: translateZ({translate_z}px);'>"
        for y in range(size):
            for x in range(size):
                val = volume[z, y, x]
                is_active = show_active and (az <= z <= az + 1) and (ay <= y <= ay + 1) and (ax <= x <= ax + 1)
                
                intensity = 0.1 + 0.8 * (abs(val) / max(1, max_val))
                bg = f"rgba({color_rgba}, {intensity})"
                if is_active:
                    bd = "2px solid currentColor"
                    bg = f"rgba({color_rgba}, {min(1.0, intensity + 0.5)})"
                else:
                    bd = f"1px solid rgba({color_rgba}, 0.5)"
                    
                txt = str(val) if val != 0 else ""
                html += f"<div style='background-color: {bg}; border: {bd}; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1rem; transition: all 0.3s ease;'>{txt}</div>"
        html += "</div>"
    html += "</div></div>"
    return html

def render_4d_output(out, az, ay, ax, title, color_rgba, max_val, current_step):
    depth, size = 2, 2
    box_size = 90
    z_spacing = 40
    
    html = f"<h5 style='text-align: center; margin-bottom: 20px; font-weight: bold;'>{title}</h5>"
    html += f"""<div style="perspective: 1200px; width: 100%; display: flex; justify-content: center; padding-top: 20px; padding-bottom: 80px;">
      <div style="transform: rotateX(60deg) rotateZ(-45deg); transform-style: preserve-3d; width: {box_size}px; height: {box_size}px; position: relative;">"""
    
    for z in range(depth):
        translate_z = z * z_spacing
        html += f"<div style='position: absolute; top: 0; left: 0; width: {box_size}px; height: {box_size}px; display: grid; grid-template-columns: repeat({size}, 1fr); grid-template-rows: repeat({size}, 1fr); gap: 2px; transform: translateZ({translate_z}px);'>"
        for y in range(size):
            for x in range(size):
                val = out[z, y, x]
                c_step = z * 4 + y * 2 + x
                is_current = (z == az and y == ay and x == ax)
                
                if c_step <= current_step:
                    intensity = 0.2 + 0.8 * (abs(val) / max(1, max_val))
                    bg = f"rgba({color_rgba}, {intensity})"
                    bd = "3px solid currentColor" if is_current else f"1px solid rgba({color_rgba}, 0.8)"
                    txt = str(val)
                else:
                    bg = "transparent"
                    bd = "1px dashed rgba(128, 128, 128, 0.4)"
                    txt = ""
                html += f"<div style='background-color: {bg}; border: {bd}; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1rem; transition: all 0.3s ease;'>{txt}</div>"
        html += "</div>"
    html += "</div></div>"
    return html

# ==============================================================================
# DASHBOARD LAYOUT RENDER
# ==============================================================================
lang = st.session_state.lang
c_t = st.session_state.step_t

col_r1_1, col_r1_2, col_r1_3 = st.columns(3)
with col_r1_1:
    title1 = f"{t[lang]['vol_t0']} $t={c_t}$" if lang == 'KOR' else f"{t[lang]['vol_t0']} $t={c_t}$"
    st.markdown(render_4d_slice(input_vols[c_t], active_z, active_y, active_x, title1, "54, 162, 235", 5), unsafe_allow_html=True)
with col_r1_2:
    title2 = f"{t[lang]['vol_t1']} $t={c_t+1}$" if lang == 'KOR' else f"{t[lang]['vol_t1']} $t={c_t+1}$"
    st.markdown(render_4d_slice(input_vols[c_t+1], active_z, active_y, active_x, title2, "54, 162, 235", 5), unsafe_allow_html=True)
with col_r1_3:
    st.markdown(render_4d_slice(diff_map, active_z, active_y, active_x, t[lang]['diff_map'], "255, 159, 64", 5, show_active=False), unsafe_allow_html=True)

st.divider()

col_r2_1, col_r2_2, col_r2_3 = st.columns(3)
with col_r2_1:
    st.markdown(render_4d_slice(kernel[0], 0, 0, 0, f"Kernel $t=0$", "255, 99, 132", 1, show_active=False), unsafe_allow_html=True)
with col_r2_2:
    st.markdown(render_4d_slice(kernel[1], 0, 0, 0, f"Kernel $t=1$", "255, 99, 132", 1, show_active=False), unsafe_allow_html=True)
with col_r2_3:
    st.markdown(render_4d_output(out_vols[c_t], active_z, active_y, active_x, f"{t[lang]['out_vol']}", "75, 192, 192", np.max(np.abs(out_vols)), st.session_state.step_s), unsafe_allow_html=True)

# ==============================================================================
# 4D CALCULATION DISPLAY
# ==============================================================================
region_t0 = input_vols[c_t, active_z:active_z+2, active_y:active_y+2, active_x:active_x+2]
region_t1 = input_vols[c_t+1, active_z:active_z+2, active_y:active_y+2, active_x:active_x+2]

eq_t0 = " + ".join([f"({region_t0.flatten()[i]}×{kernel[0].flatten()[i]})" for i in range(8)])
eq_t1 = " + ".join([f"({region_t1.flatten()[i]}×{kernel[1].flatten()[i]})" for i in range(8)])
final_val = out_vols[c_t, active_z, active_y, active_x]

st.success(f"### 🧮 {t[lang]['calc']}\n\n**Spatial Sum at $t={c_t}$:** `[ {eq_t0} ]`\n\n**Spatial Sum at $t={c_t+1}$:** `[ {eq_t1} ]`\n\n**Total 4D Output = Time $t={c_t}$ Sum + Time $t={c_t+1}$ Sum = {final_val}**")

# ==============================================================================
# AUTO-PLAY LOGIC
# ==============================================================================
if st.session_state.autoplay_4d:
    time.sleep(1.0)
    st.session_state.step_s += 1
    if st.session_state.step_s > 7:
        st.session_state.step_s = 0
        st.session_state.step_t += 1
        if st.session_state.step_t > 3:
            st.session_state.step_t = 0
            st.session_state.autoplay_4d = False
    st.rerun()
