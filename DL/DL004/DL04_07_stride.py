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
        'page_title': "Convolution Stride Visualizer",
        'title': "Convolution Stride: Stride 1 vs Stride 2",
        'subtitle': "An interactive animation demonstrating the mechanical process and output shapes of a 2D convolution when using different step sizes (stride).",
        'controls_header': "### Controls",
        'lbl_stride': "Stride Size",
        'opt_stride1': "Stride = 1 (1 pixel jump)",
        'opt_stride2': "Stride = 2 (2 pixel jump)",
        'btn_next': "Next Step",
        'btn_reset': "Reset",
        'toggle_autoplay': "Auto Play",
        'input_matrix': "Input ({r}x{c})",
        'filter_matrix': "Filter (3x3)",
        'output_matrix': "Output ({r}x{c})",
        'formula_header': "### Size Calculation Formula",
        'formula_base': r"\text{Output Size} = \lfloor \frac{\text{Input} - \text{Kernel} + (2 \times \text{Padding})}{\text{Stride}} \rfloor + 1",
        'theory_title': "Theory: Concept of Stride (Pros & Cons)",
        'theory_content': r"""### What is Stride?
Stride refers to the number of pixels the sliding filter shifts across the input matrix at each step. 

### Stride = 1 (Default)
* **Detailed Feature Extraction:** Moving 1 pixel at a time ensures that the filter densely samples every possible overlapping region.
* **Large Output Size:** Produces a large output feature map, retaining a high amount of spatial information.
* **Cons:** High computational cost and high memory usage because the spatial dimensions do not reduce quickly.

### Stride > 1 (e.g., Stride 2)
* **Downsampling:** Skipping pixels rapidly shrinks the output spatial dimensions (e.g., a 6x6 input becomes a 2x2 output with stride 2).
* **Efficiency:** Drastically reduces the number of mathematical operations and memory footprint, allowing for deeper networks.
* **Pooling Effect:** Acts similarly to a pooling layer by aggregating spatial information.
* **Cons:** May miss fine-grained local details or patterns since the filter jumps over intermediate regions."""
    },
    'KOR': {
        'page_title': "합성곱 스트라이드 시각화 도구",
        'title': "합성곱 스트라이드: Stride 1 vs Stride 2",
        'subtitle': "필터의 이동 간격(스트라이드)에 따른 2D 합성곱 처리 과정과 출력 형태의 차이를 비교하는 인터랙티브 애니메이션입니다.",
        'controls_header': "### 제어",
        'lbl_stride': "스트라이드(Stride) 크기",
        'opt_stride1': "Stride = 1 (1칸씩 이동)",
        'opt_stride2': "Stride = 2 (2칸씩 이동)",
        'btn_next': "다음 단계",
        'btn_reset': "초기화",
        'toggle_autoplay': "자동 재생",
        'input_matrix': "입력 ({r}x{c})",
        'filter_matrix': "필터 (3x3)",
        'output_matrix': "출력 ({r}x{c})",
        'formula_header': "### 출력 크기 계산",
        'formula_base': r"\text{출력 크기} = \lfloor \frac{\text{입력} - \text{커널} + (2 \times \text{패딩})}{\text{스트라이드}} \rfloor + 1",
        'theory_title': "이론: 스트라이드(Stride)의 개념과 장단점",
        'theory_content': r"""### 스트라이드(Stride)란?
스트라이드는 슬라이딩 필터가 입력 행렬 위를 이동할 때 한 번에 몇 픽셀씩 건너뛸지를 결정하는 이동 간격(보폭)입니다.

### Stride = 1 (기본값)
* **세밀한 특성 추출:** 한 번에 1픽셀씩 촘촘하게 이동하므로 가능한 모든 겹치는 영역을 분석합니다.
* **큰 출력 크기:** 출력 특성 맵의 크기가 크며, 공간적인 정보를 많이 보존합니다.
* **단점:** 연산량이 많고 메모리 사용량이 높습니다.

### Stride > 1 (예: Stride 2)
* **다운샘플링(Downsampling):** 픽셀을 건너뛰며 이동하여 출력의 공간적 차원을 빠르게 축소합니다 (예: 6x6 입력이 2x2 출력이 됨).
* **효율성:** 총 연산 횟수와 메모리 요구량을 획기적으로 줄여, 네트워크를 더 깊게 쌓을 수 있도록 도와줍니다.
* **풀링(Pooling) 효과:** 공간적 정보를 집약시켜 풀링 계층과 유사한 역할을 수행합니다.
* **단점:** 중간 영역을 건너뛰기 때문에 미세하고 세밀한 지역적 패턴(디테일)을 놓칠 가능성이 있습니다."""
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
if 'str_mode' not in st.session_state:
    st.session_state.str_mode = 'opt_stride1'
if 'str_step' not in st.session_state:
    st.session_state.str_step = 0
if 'str_autoplay' not in st.session_state:
    st.session_state.str_autoplay = False

# ==============================================================================
# CONTROLS
# ==============================================================================
col_ctrl1, col_ctrl2, col_ctrl3, _ = st.columns([1.5, 1.5, 1.5, 3])

with col_ctrl1:
    str_options = ['opt_stride1', 'opt_stride2']
    str_labels = [t[st.session_state.lang][opt] for opt in str_options]
    curr_idx = str_options.index(st.session_state.str_mode)
    
    sel_label = st.selectbox(t[st.session_state.lang]['lbl_stride'], str_labels, index=curr_idx)
    sel_mode = str_options[str_labels.index(sel_label)]

    if sel_mode != st.session_state.str_mode:
        st.session_state.str_mode = sel_mode
        st.session_state.str_step = 0
        st.session_state.str_autoplay = False
        st.rerun()

stride_val = 1 if st.session_state.str_mode == 'opt_stride1' else 2
out_dim = ((6 - 3) // stride_val) + 1
max_steps = out_dim * out_dim

with col_ctrl2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    if st.button(t[st.session_state.lang]['btn_next'], disabled=st.session_state.str_step >= max_steps - 1 or st.session_state.str_autoplay, use_container_width=True):
        st.session_state.str_step += 1
        st.rerun()
    if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
        st.session_state.str_step = 0
        st.session_state.str_autoplay = False
        st.rerun()

with col_ctrl3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    auto_play = st.toggle(t[st.session_state.lang]['toggle_autoplay'], value=st.session_state.str_autoplay)
    if auto_play != st.session_state.str_autoplay:
        st.session_state.str_autoplay = auto_play
        st.rerun()

st.divider()

# ==============================================================================
# MATRIX DATA GENERATION
# ==============================================================================
# 6x6 Input Grid
input_grid = np.array([
    [1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 1],
    [1, 2, 3, 3, 2, 1],
    [1, 2, 3, 3, 2, 1],
    [1, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1]
])

# 3x3 Filter
filt = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
])

output_grid = np.zeros((out_dim, out_dim), dtype=int)

# Pre-compute up to current step
curr_out_r = st.session_state.str_step // out_dim
curr_out_c = st.session_state.str_step % out_dim

for r in range(out_dim):
    for c in range(out_dim):
        if r * out_dim + c <= st.session_state.str_step:
            in_r = r * stride_val
            in_c = c * stride_val
            region = input_grid[in_r:in_r+3, in_c:in_c+3]
            output_grid[r, c] = np.sum(region * filt)

# ==============================================================================
# HTML RENDERING
# ==============================================================================
def render_matrix_html(matrix, name, step, stride, is_input=False, is_filter=False, is_output=False):
    html = f"<h4 style='text-align: center; margin-top: 10px;'>{name}</h4>"
    html += "<table style='border-collapse: collapse; margin: 0 auto; font-size: 1.2rem; font-family: monospace;'>"
    
    rows, cols = matrix.shape
    
    dim_out = ((6 - 3) // stride) + 1
    out_r = step // dim_out
    out_c = step % dim_out
    
    in_start_r = out_r * stride
    in_start_c = out_c * stride
    
    for i in range(rows):
        html += "<tr>"
        for j in range(cols):
            val = matrix[i, j]
            
            border = "1px solid currentColor"
            font_weight = "normal"
            opacity = "1.0"
            bg_color = "transparent"
                    
            # Highlight active regions
            if is_input:
                if in_start_r <= i < in_start_r + 3 and in_start_c <= j < in_start_c + 3:
                    border = "3px solid currentColor"
                    font_weight = "bold"
                    bg_color = "rgba(128, 128, 128, 0.2)"
            elif is_output:
                if i == out_r and j == out_c:
                    border = "3px solid currentColor"
                    font_weight = "bold"
                    bg_color = "rgba(128, 128, 128, 0.2)"
                elif i * dim_out + j > step:
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

if stride_val == 1:
    st.latex(r"= \lfloor \frac{6 - 3 + (2 \times 0)}{1} \rfloor + 1 = \mathbf{4 \times 4}")
    st.info("Total steps (Windows processed): **16**")
else:
    st.latex(r"= \lfloor \frac{6 - 3 + (2 \times 0)}{2} \rfloor + 1 = \mathbf{2 \times 2}")
    st.info("Total steps (Windows processed): **4**")
    
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

col_in, col_filt, col_out = st.columns([2, 1, 2])

in_r, in_c = input_grid.shape
out_r, out_c = output_grid.shape

str_in = t[st.session_state.lang]['input_matrix'].format(r=in_r, c=in_c)
str_filt = t[st.session_state.lang]['filter_matrix']
str_out = t[st.session_state.lang]['output_matrix'].format(r=out_r, c=out_c)

with col_in:
    st.markdown(render_matrix_html(input_grid, str_in, st.session_state.str_step, stride_val, is_input=True), unsafe_allow_html=True)
with col_filt:
    st.markdown(render_matrix_html(filt, str_filt, st.session_state.str_step, stride_val, is_filter=True), unsafe_allow_html=True)
with col_out:
    st.markdown(render_matrix_html(output_grid, str_out, st.session_state.str_step, stride_val, is_output=True), unsafe_allow_html=True)

# Auto-play logic
if st.session_state.str_autoplay and st.session_state.str_step < max_steps - 1:
    time.sleep(0.6)
    st.session_state.str_step += 1
    st.rerun()
elif st.session_state.str_autoplay and st.session_state.str_step >= max_steps - 1:
    st.session_state.str_autoplay = False
    st.rerun()
