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
        'page_title': "3D Neural Network Visualizer",
        'title': "3D CNN: Spatial Voxel Convolution",
        'subtitle': "An interactive visualizer demonstrating how a 3D kernel processes a volumetric (Voxel) grid.",
        'btn_step': "▶ Step Traversal",
        'btn_reset': "Reset",
        'toggle_autoplay': "Auto-Play",
        'toggle_view': "View Mode",
        'opt_voxel': "Voxel View (Solid)",
        'opt_math': "Mathematical View (Numeric)",
        'info_title': "Educational Overlay: 3D Convolutions",
        'info_1': "### 1. What is a Voxel?\nA **Voxel** (Volumetric Pixel) is a 3D unit of data, extending the concept of a 2D pixel into depth. It represents a single value on a regular grid in 3D space.",
        'info_2': "### 2. Kernel Depth\nIn a **3D Convolution**, the kernel is a 3D volume (e.g., 2x2x2). It must have depth to capture spatial relationships along the Z-axis, just as it does along the X and Y axes.",
        'info_3': "### 3. 2D vs 3D Convolutions\n- **2D Convolution:** Slides in 2 directions (X, Y). Used for standard images.\n- **3D Convolution:** Slides in 3 directions (X, Y, Z), producing a 3D output volume. Commonly used for MRI/CT scans or video frame sequences.",
        'title_in': "Input Volume (3x3x3)",
        'title_ker': "3D Kernel (2x2x2)",
        'title_out': "Output Volume (2x2x2)",
        'calc': "Calculation"
    },
    'KOR': {
        'page_title': "3D 신경망 학습 시각화 도구",
        'title': "3D CNN: 공간적 복셀(Voxel) 합성곱",
        'subtitle': "3D 커널이 3차원 부피(Voxel) 그리드를 어떻게 슬라이딩하며 처리하는지 보여주는 대화형 시각화 도구입니다.",
        'btn_step': "▶ 단계 진행 (Step)",
        'btn_reset': "초기화",
        'toggle_autoplay': "자동 재생",
        'toggle_view': "뷰 모드 (View Mode)",
        'opt_voxel': "복셀 뷰 (Voxel View)",
        'opt_math': "수식 뷰 (Math View)",
        'info_title': "이론: 3D 합성곱 (3D Convolutions)",
        'info_1': "### 1. 복셀(Voxel)이란?\n**복셀(Volumetric Pixel)**은 2D 픽셀의 개념을 3차원으로 확장한 단위입니다. 3D 공간상의 격자에서 단일 데이터 값을 나타냅니다.",
        'info_2': "### 2. 커널의 깊이 (Depth)\n**3D 합성곱**에서 커널(필터)은 3D 부피(예: 2x2x2)를 가집니다. 가로(X)와 세로(Y)뿐만 아니라 깊이(Z) 축을 따라서도 공간적 관계를 파악하기 위해 깊이 방향으로 슬라이딩해야 합니다.",
        'info_3': "### 3. 2D vs 3D 합성곱\n- **2D 합성곱:** X, Y 두 방향으로만 이동하며 2D 결과를 만듭니다. (일반 이미지)\n- **3D 합성곱:** X, Y, Z 세 방향으로 이동하며 3D 결과를 생성합니다. (의료용 MRI/CT 스캔, 비디오 등 공간/시간 데이터)",
        'title_in': "입력 부피 (3x3x3)",
        'title_ker': "3D 커널 (2x2x2)",
        'title_out': "출력 부피 (2x2x2)",
        'calc': "계산 과정"
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
if 'step_3d' not in st.session_state:
    st.session_state.step_3d = 0
if 'autoplay_3d' not in st.session_state:
    st.session_state.autoplay_3d = False
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'opt_math'

# ==============================================================================
# CONTROLS
# ==============================================================================
col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([1.5, 1.5, 2, 4])

max_steps = 8 # 2x2x2 output = 8 steps

with col_ctrl1:
    if st.button(t[st.session_state.lang]['btn_step'], disabled=st.session_state.autoplay_3d, use_container_width=True):
        if st.session_state.step_3d >= max_steps - 1:
            st.session_state.step_3d = 0
        else:
            st.session_state.step_3d += 1
        st.rerun()

with col_ctrl2:
    if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
        st.session_state.step_3d = 0
        st.session_state.autoplay_3d = False
        st.rerun()

with col_ctrl3:
    auto_play = st.toggle(t[st.session_state.lang]['toggle_autoplay'], value=st.session_state.autoplay_3d)
    if auto_play != st.session_state.autoplay_3d:
        st.session_state.autoplay_3d = auto_play
        st.rerun()

with col_ctrl4:
    view_opts = ['opt_voxel', 'opt_math']
    view_labels = [t[st.session_state.lang][opt] for opt in view_opts]
    curr_view_idx = view_opts.index(st.session_state.view_mode)
    
    sel_view_label = st.radio(t[st.session_state.lang]['toggle_view'], view_labels, index=curr_view_idx, horizontal=True, label_visibility="collapsed")
    sel_view_mode = view_opts[view_labels.index(sel_view_label)]
    
    if sel_view_mode != st.session_state.view_mode:
        st.session_state.view_mode = sel_view_mode
        st.rerun()

st.divider()

# ==============================================================================
# 3D DATA GENERATION
# ==============================================================================
# 3x3x3 Input Volume
input_vol = np.array([
    [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
])

# 2x2x2 Kernel
kernel = np.array([
    [[1, -1], [0, 1]],
    [[1, 0], [-1, 1]]
])

# Pre-calculate output
out_vol = np.zeros((2, 2, 2), dtype=int)
for z in range(2):
    for y in range(2):
        for x in range(2):
            region = input_vol[z:z+2, y:y+2, x:x+2]
            out_vol[z, y, x] = np.sum(region * kernel)

# Current step indices
active_z = st.session_state.step_3d // 4
active_y = (st.session_state.step_3d % 4) // 2
active_x = st.session_state.step_3d % 2

# ==============================================================================
# 3D ISOMETRIC RENDERING FUNCTIONS
# ==============================================================================
def render_3d_input(volume, az, ay, ax, v_mode, title):
    html = f"<h4 style='text-align: center; margin-bottom: 30px;'>{title}</h4>"
    html += """<div style="perspective: 1200px; width: 200px; margin: 0 auto; padding-top: 30px; padding-bottom: 120px;">
      <div style="transform: rotateX(60deg) rotateZ(-45deg); transform-style: preserve-3d; width: 150px; height: 150px; position: relative;">"""
    
    for z in range(3):
        translate_z = z * 60
        html += f"<div style='position: absolute; top: 0; left: 0; width: 150px; height: 150px; display: grid; grid-template-columns: repeat(3, 1fr); grid-template-rows: repeat(3, 1fr); gap: 3px; transform: translateZ({translate_z}px);'>"
        for y in range(3):
            for x in range(3):
                val = volume[z, y, x]
                is_active = (az <= z <= az + 1) and (ay <= y <= ay + 1) and (ax <= x <= ax + 1)
                
                if v_mode == 'opt_voxel':
                    bg = "rgba(54, 162, 235, 0.85)" if is_active else "rgba(128, 128, 128, 0.3)"
                    bd = "2px solid rgba(255, 255, 255, 0.6)" if is_active else "1px solid rgba(128, 128, 128, 0.5)"
                    txt = ""
                else:
                    bg = "rgba(54, 162, 235, 0.4)" if is_active else "rgba(128, 128, 128, 0.1)"
                    bd = "2px solid currentColor" if is_active else "1px solid rgba(128, 128, 128, 0.4)"
                    txt = str(val)
                
                html += f"<div style='background-color: {bg}; border: {bd}; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem; transition: all 0.3s ease;'>{txt}</div>"
        html += "</div>"
    html += "</div></div>"
    return html

def render_3d_kernel(ker, v_mode, title):
    html = f"<h4 style='text-align: center; margin-bottom: 30px;'>{title}</h4>"
    html += """<div style="perspective: 1200px; width: 150px; margin: 0 auto; padding-top: 30px; padding-bottom: 120px;">
      <div style="transform: rotateX(60deg) rotateZ(-45deg); transform-style: preserve-3d; width: 100px; height: 100px; position: relative;">"""
    
    for z in range(2):
        translate_z = z * 60
        html += f"<div style='position: absolute; top: 0; left: 0; width: 100px; height: 100px; display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); gap: 3px; transform: translateZ({translate_z}px);'>"
        for y in range(2):
            for x in range(2):
                val = ker[z, y, x]
                if v_mode == 'opt_voxel':
                    bg = "rgba(255, 99, 132, 0.85)"
                    bd = "2px solid rgba(255, 255, 255, 0.6)"
                    txt = ""
                else:
                    bg = "rgba(255, 99, 132, 0.4)"
                    bd = "2px solid currentColor"
                    txt = str(val)
                html += f"<div style='background-color: {bg}; border: {bd}; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem; transition: all 0.3s ease;'>{txt}</div>"
        html += "</div>"
    html += "</div></div>"
    return html

def render_3d_output(out, az, ay, ax, v_mode, title, step):
    html = f"<h4 style='text-align: center; margin-bottom: 30px;'>{title}</h4>"
    html += """<div style="perspective: 1200px; width: 150px; margin: 0 auto; padding-top: 30px; padding-bottom: 120px;">
      <div style="transform: rotateX(60deg) rotateZ(-45deg); transform-style: preserve-3d; width: 100px; height: 100px; position: relative;">"""
    
    for z in range(2):
        translate_z = z * 60
        html += f"<div style='position: absolute; top: 0; left: 0; width: 100px; height: 100px; display: grid; grid-template-columns: repeat(2, 1fr); grid-template-rows: repeat(2, 1fr); gap: 3px; transform: translateZ({translate_z}px);'>"
        for y in range(2):
            for x in range(2):
                val = out[z, y, x]
                c_step = z * 4 + y * 2 + x
                is_current = (z == az and y == ay and x == ax)
                
                if c_step <= step:
                    if v_mode == 'opt_voxel':
                        bg = "rgba(75, 192, 192, 0.85)" if is_current else "rgba(75, 192, 192, 0.4)"
                        bd = "2px solid rgba(255, 255, 255, 0.8)" if is_current else "1px solid rgba(255, 255, 255, 0.4)"
                        txt = ""
                    else:
                        bg = "rgba(75, 192, 192, 0.5)" if is_current else "rgba(75, 192, 192, 0.2)"
                        bd = "3px solid currentColor" if is_current else "1px solid currentColor"
                        txt = str(val)
                else:
                    bg = "transparent"
                    bd = "1px dashed rgba(128, 128, 128, 0.4)"
                    txt = ""
                    
                html += f"<div style='background-color: {bg}; border: {bd}; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 1.2rem; transition: all 0.3s ease;'>{txt}</div>"
        html += "</div>"
    html += "</div></div>"
    return html

# ==============================================================================
# MAIN RENDER
# ==============================================================================
st.info(f"**Current Status:** Z={active_z}, Y={active_y}, X={active_x}  (Step {st.session_state.step_3d + 1}/{max_steps})")

col_in, col_ker, col_out = st.columns([2, 1, 2])

with col_in:
    st.markdown(render_3d_input(input_vol, active_z, active_y, active_x, st.session_state.view_mode, t[st.session_state.lang]['title_in']), unsafe_allow_html=True)
    
with col_ker:
    st.markdown(render_3d_kernel(kernel, st.session_state.view_mode, t[st.session_state.lang]['title_ker']), unsafe_allow_html=True)
    
with col_out:
    st.markdown(render_3d_output(out_vol, active_z, active_y, active_x, st.session_state.view_mode, t[st.session_state.lang]['title_out'], st.session_state.step_3d), unsafe_allow_html=True)

# Calculation Display
active_region = input_vol[active_z:active_z+2, active_y:active_y+2, active_x:active_x+2]
eq_terms = [f"({active_region.flatten()[i]}×{kernel.flatten()[i]})" for i in range(8)]
eq_str = " + ".join(eq_terms)
final_val = out_vol[active_z, active_y, active_x]

st.success(f"### {t[st.session_state.lang]['calc']}\n\n**Output Value:** {eq_str} = **{final_val}**")

# ==============================================================================
# AUTO-PLAY LOGIC
# ==============================================================================
if st.session_state.autoplay_3d and st.session_state.step_3d < max_steps - 1:
    time.sleep(1.2)
    st.session_state.step_3d += 1
    st.rerun()
elif st.session_state.autoplay_3d and st.session_state.step_3d >= max_steps - 1:
    st.session_state.autoplay_3d = False
    st.rerun()
