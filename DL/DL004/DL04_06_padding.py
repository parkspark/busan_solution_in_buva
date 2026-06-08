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
        'page_title': "Convolution Padding Visualizer",
        'title': "Convolution Padding: Valid vs Same",
        'subtitle': "An interactive animation comparing the mechanical process and output shapes of a 2D convolution using 'valid' and 'same' padding.",
        'controls_header': "### Controls",
        'lbl_pad': "Padding Mode",
        'opt_valid': "Valid (No Padding)",
        'opt_same': "Same (Zero Padding)",
        'btn_next': "Next Step",
        'btn_reset': "Reset",
        'toggle_autoplay': "Auto Play",
        'input_matrix': "Input ({r}x{c})",
        'filter_matrix': "Filter (3x3)",
        'output_matrix': "Output ({r}x{c})",
        'formula_header': "### Size Calculation Formula",
        'formula_base': r"\text{Output Size} = \text{Input} - \text{Kernel} + 1 + (2 \times \text{Padding})",
        'theory_title': "Theory: Concept of Padding (Pros & Cons)",
        'theory_content': r"""### What is Padding?
Padding refers to the process of adding extra layers of pixels (usually zeros) around the borders of an input image or feature map before applying a convolution filter. 

### Pros of using Padding ('Same' Padding)
* **Preserves Spatial Dimensions:** By adding padding, the output feature map can retain the exact same width and height as the original input.
* **Prevents Information Loss at Edges:** Without padding, pixels at the very edge of an image are only processed once or twice by the sliding filter, whereas central pixels are processed many times. Padding ensures edge pixels contribute more equally to the output.
* **Enables Deeper Networks:** By preventing the feature map from shrinking after every convolution layer, we can build much deeper networks without the spatial dimensions collapsing to zero.

### Cons of using Padding
* **Increased Computation:** The addition of padded pixels increases the total spatial area, which means the convolution operation must perform more calculations.
* **Artificial Edges:** Zero-padding introduces artificial sudden drops in pixel intensity at the borders (e.g., from a bright pixel to a padded `0`), which the filter might mistakenly interpret as a real edge/feature."""
    },
    'KOR': {
        'page_title': "합성곱 패딩 시각화 도구",
        'title': "합성곱 패딩: Valid vs Same",
        'subtitle': "'valid' 패딩과 'same' 패딩을 사용할 때의 2D 합성곱 처리 과정과 출력 형태를 비교하는 인터랙티브 애니메이션입니다.",
        'controls_header': "### 제어",
        'lbl_pad': "패딩 모드",
        'opt_valid': "Valid (패딩 없음)",
        'opt_same': "Same (제로 패딩)",
        'btn_next': "다음 단계",
        'btn_reset': "초기화",
        'toggle_autoplay': "자동 재생",
        'input_matrix': "입력 ({r}x{c})",
        'filter_matrix': "필터 (3x3)",
        'output_matrix': "출력 ({r}x{c})",
        'formula_header': "### 출력 크기 계산",
        'formula_base': r"\text{출력 크기} = \text{입력} - \text{커널} + 1 + (2 \times \text{패딩})",
        'theory_title': "이론: 패딩(Padding)의 개념과 장단점",
        'theory_content': r"""### 패딩(Padding)이란?
패딩은 합성곱(Convolution) 필터를 적용하기 전에 입력 이미지나 특성 맵의 가장자리에 추가적인 픽셀 테두리(일반적으로 0)를 덧붙이는 과정입니다.

### 패딩 사용의 장점 ('Same' 패딩)
* **공간적 차원 유지:** 패딩을 추가하면 출력되는 특성 맵의 크기를 원본 입력과 완전히 동일하게 유지할 수 있습니다.
* **가장자리 정보 유실 방지:** 패딩이 없으면 이미지 중앙의 픽셀은 필터에 여러 번 겹쳐 처리되지만, 가장자리 픽셀은 한두 번만 처리됩니다. 패딩을 사용하면 가장자리 정보도 출력에 충분히 기여하게 됩니다.
* **더 깊은 신경망 구축 가능:** 합성곱 계층을 통과할 때마다 특성 맵의 크기가 축소되는 것을 막아주어, 이미지의 공간 정보가 완전히 사라지기 전까지 훨씬 더 깊은 신경망을 설계할 수 있게 해줍니다.

### 패딩 사용의 단점
* **계산량 증가:** 패딩 픽셀이 추가됨에 따라 전체 면적이 커지며, 필터가 슬라이딩하며 수행해야 하는 전체 연산 횟수도 증가합니다.
* **인공적인 경계(Noise) 생성:** 제로 패딩(Zero-Padding)의 경우, 이미지 가장자리의 값이 갑자기 0으로 뚝 떨어지는 현상이 생깁니다. 필터는 이러한 인위적인 밝기 변화를 실제 이미지의 윤곽선(에지)으로 착각하여 불필요한 패턴을 학습할 위험이 있습니다."""
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
# THEORY EXPLANATION
# ==============================================================================
with st.expander(t[st.session_state.lang]['theory_title']):
    st.markdown(t[st.session_state.lang]['theory_content'])

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================
if 'pad_mode' not in st.session_state:
    st.session_state.pad_mode = 'opt_valid'
if 'pad_step' not in st.session_state:
    st.session_state.pad_step = 0
if 'pad_autoplay' not in st.session_state:
    st.session_state.pad_autoplay = False

# ==============================================================================
# CONTROLS
# ==============================================================================
col_ctrl1, col_ctrl2, col_ctrl3, _ = st.columns([1.5, 1.5, 1.5, 3])

with col_ctrl1:
    pad_options = ['opt_valid', 'opt_same']
    pad_labels = [t[st.session_state.lang][opt] for opt in pad_options]
    curr_idx = pad_options.index(st.session_state.pad_mode)
    
    sel_label = st.selectbox(t[st.session_state.lang]['lbl_pad'], pad_labels, index=curr_idx)
    sel_mode = pad_options[pad_labels.index(sel_label)]

    if sel_mode != st.session_state.pad_mode:
        st.session_state.pad_mode = sel_mode
        st.session_state.pad_step = 0
        st.session_state.pad_autoplay = False
        st.rerun()

max_steps = 9 if st.session_state.pad_mode == 'opt_valid' else 25

with col_ctrl2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button(t[st.session_state.lang]['btn_next'], disabled=st.session_state.pad_step >= max_steps - 1 or st.session_state.pad_autoplay, use_container_width=True):
        st.session_state.pad_step += 1
        st.rerun()
    if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
        st.session_state.pad_step = 0
        st.session_state.pad_autoplay = False
        st.rerun()

with col_ctrl3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    auto_play = st.toggle(t[st.session_state.lang]['toggle_autoplay'], value=st.session_state.pad_autoplay)
    if auto_play != st.session_state.pad_autoplay:
        st.session_state.pad_autoplay = auto_play
        st.rerun()

st.divider()

# ==============================================================================
# MATRIX DATA GENERATION
# ==============================================================================
base_input = np.array([
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],
    [1, 2, 3, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1]
])

filt = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
])

if st.session_state.pad_mode == 'opt_same':
    input_grid = np.pad(base_input, 1, mode='constant', constant_values=0)
    out_dim = 5
else:
    input_grid = base_input.copy()
    out_dim = 3

output_grid = np.zeros((out_dim, out_dim), dtype=int)

# Pre-compute up to current step
out_row = st.session_state.pad_step // out_dim
out_col = st.session_state.pad_step % out_dim

for r in range(out_dim):
    for c in range(out_dim):
        if r * out_dim + c <= st.session_state.pad_step:
            region = input_grid[r:r+3, c:c+3]
            output_grid[r, c] = np.sum(region * filt)

# ==============================================================================
# SVG / HTML RENDERING
# ==============================================================================
def render_matrix_html(matrix, name, step, pad_mode, is_input=False, is_filter=False, is_output=False):
    html = f"<h4 style='text-align: center; margin-top: 10px;'>{name}</h4>"
    html += "<table style='border-collapse: collapse; margin: 0 auto; font-size: 1.2rem; font-family: monospace;'>"
    
    rows, cols = matrix.shape
    
    out_cols = 3 if pad_mode == 'opt_valid' else 5
    curr_r = step // out_cols
    curr_c = step % out_cols
    
    for i in range(rows):
        html += "<tr>"
        for j in range(cols):
            val = matrix[i, j]
            
            border = "1px solid currentColor"
            font_weight = "normal"
            opacity = "1.0"
            bg_color = "transparent"
            
            # Functional styling for zeros in SAME padding
            if is_input and pad_mode == 'opt_same':
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    border = "1px dashed currentColor"
                    opacity = "0.5"
                    
            # Highlight active regions
            if is_input:
                if curr_r <= i < curr_r + 3 and curr_c <= j < curr_c + 3:
                    border = "3px solid currentColor"
                    font_weight = "bold"
                    bg_color = "rgba(128, 128, 128, 0.2)"
            elif is_output:
                if i == curr_r and j == curr_c:
                    border = "3px solid currentColor"
                    font_weight = "bold"
                    bg_color = "rgba(128, 128, 128, 0.2)"
                elif i * out_cols + j > step:
                    val = "" 
                    border = "1px dotted currentColor"
                    opacity = "0.3"
            elif is_filter:
                border = "1px solid currentColor"
                
            html += f"<td style='width: 45px; height: 45px; text-align: center; vertical-align: middle; border: {border}; font-weight: {font_weight}; opacity: {opacity}; background-color: {bg_color};'>{val}</td>"
            
        html += "</tr>"
    html += "</table>"
    return html

# ==============================================================================
# MAIN RENDER
# ==============================================================================
st.markdown(t[st.session_state.lang]['formula_header'])
st.latex(t[st.session_state.lang]['formula_base'])

if st.session_state.pad_mode == 'opt_valid':
    st.latex(r"= 5 - 3 + 1 + (2 \times \mathbf{0}) = \mathbf{3 \times 3}")
else:
    st.latex(r"= 5 - 3 + 1 + (2 \times \mathbf{1}) = \mathbf{5 \times 5}")
    
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

col_in, col_filt, col_out = st.columns([2, 1, 2])

in_r, in_c = input_grid.shape
out_r, out_c = output_grid.shape

str_in = t[st.session_state.lang]['input_matrix'].format(r=in_r, c=in_c)
str_filt = t[st.session_state.lang]['filter_matrix']
str_out = t[st.session_state.lang]['output_matrix'].format(r=out_r, c=out_c)

with col_in:
    st.markdown(render_matrix_html(input_grid, str_in, st.session_state.pad_step, st.session_state.pad_mode, is_input=True), unsafe_allow_html=True)
with col_filt:
    st.markdown(render_matrix_html(filt, str_filt, st.session_state.pad_step, st.session_state.pad_mode, is_filter=True), unsafe_allow_html=True)
with col_out:
    st.markdown(render_matrix_html(output_grid, str_out, st.session_state.pad_step, st.session_state.pad_mode, is_output=True), unsafe_allow_html=True)

# Auto-play logic
if st.session_state.pad_autoplay and st.session_state.pad_step < max_steps - 1:
    time.sleep(0.5)
    st.session_state.pad_step += 1
    st.rerun()
elif st.session_state.pad_autoplay and st.session_state.pad_step >= max_steps - 1:
    st.session_state.pad_autoplay = False
    st.rerun()
