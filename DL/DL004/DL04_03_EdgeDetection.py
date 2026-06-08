import streamlit as st
import numpy as np
import time
import os
from PIL import Image

# ==============================================================================
# INITIALIZE SESSION STATE & LANGUAGE
# ==============================================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'KOR'

t = {
    'ENG': {
        'page_title': "Edge Detection Visualizer",
        'title': "Edge Detection Visualizer",
        'subtitle': "This interactive visualizer demonstrates how vertical and horizontal kernels process images to detect edges.",
        'controls_header': "### Controls",
        'lbl_input': "Input Image Pattern",
        'lbl_filter': "Filter Type",
        'opt_input_vert': "Vertical Stripe",
        'opt_input_horiz': "Horizontal Stripe",
        'opt_input_flat': "Flat Image",
        'opt_filter_vert': "Vertical Edge Filter",
        'opt_filter_horiz': "Horizontal Edge Filter",
        'btn_next': "Next Step",
        'btn_reset': "Reset",
        'toggle_autoplay': "Auto Play",
        'input_matrix': "Input ({r}x{c})",
        'filter_matrix': "Filter ({r}x{c})",
        'output_matrix': "Output ({r}x{c})",
        'calc_header': "### Calculation Details",
        'output_label': "\\text{Output}",
        'complete_msg': "Edge detection scan complete!",
        'tab_matrix': "Matrix Animation",
        'tab_image': "Real Image Upload",
        'upload_prompt': "Upload an image (PNG, JPG, JPEG)",
        'original_image': "Original Image (Grayscale)",
        'filtered_image': "Edge Detected Image",
        'opt_filter_both': "Magnitude (Both)",
        'theory_title': "Theory: Convolution & Edge Detection",
        'theory_content': r"""### 1. The Core Mechanism: Convolution
The primary engine of a CNN is the convolution operation. You can think of this as shining a small flashlight over an image, section by section.
The "flashlight" is called a **Kernel** (or Filter). A kernel is a small matrix of numbers (weights). Typically, in the first layer of a CNN, these are $3 \times 3$ matrices. The network slides this kernel across the entire input image, from left to right, top to bottom.

At each step, it performs an element-wise multiplication between the kernel's weights and the pixel values of the image it is currently hovering over. It then sums all those products together to produce a single new number. This process generates a new grid called a **Feature Map**.

### 2. The Math of an Edge
How does a matrix of numbers actually find an edge? An edge in an image is simply a location where there is a sharp transition in pixel intensity (e.g., going from black to white).

Specific patterns of numbers in a kernel act as detectors for these transitions. A classic example is the **Sobel Filter**, which CNNs frequently mimic in their early layers.

**Vertical Edge Filter:**
$$
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$
If you slide this filter over a uniform area of an image (all white or all black), the negative numbers on the left will perfectly cancel out the positive numbers on the right, resulting in a sum of $0$. However, if you slide it over a transition from dark to light, the positive side multiplies against higher values while the negative side multiplies against lower values. The sum will be a large non-zero number, mathematically "lighting up" that pixel on the resulting Feature Map to indicate a vertical edge.

**Horizontal Edge Filter:**
$$
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$
This filter applies the exact same logic but is rotated to detect sharp transitions in intensity from top to bottom.

### 3. How CNNs Implement This
While classic algorithms require humans to hand-code the $-1$, $0$, and $1$ values, CNNs take a different approach:
* **Random Initialization:** When a CNN is first created, the values inside its filters are set to small, random numbers. It doesn't know what an edge is.
* **Forward Pass:** The image is passed through these random filters, producing chaotic, meaningless feature maps.
* **Loss Calculation:** The network makes a prediction (e.g., "Is this a dog?") and compares it to the correct answer to calculate its error (the loss).
* **Backpropagation:** The network uses calculus to figure out how to adjust the numbers inside its filters to make the error smaller next time.

Over thousands of iterations, the CNN naturally figures out that detecting edges is the most mathematically efficient way to start understanding shapes. By the time training is complete, if you look inside the filters of the very first layer of a trained CNN, you will find matrices that look almost exactly like the Sobel filters above—perfectly tuned edge detectors.
"""
    },
    'KOR': {
        'page_title': "에지 감지(Edge Detection)",
        'title': "에지 감지(Edge Detection)",
        'subtitle': "수직 및 수평 커널이 이미지를 처리하여 윤곽선(Edge)을 감지하는 과정을 보여줍니다.",
        'controls_header': "### 제어",
        'lbl_input': "입력 이미지 패턴",
        'lbl_filter': "필터 종류",
        'opt_input_vert': "수직 줄무늬 (Vertical Stripe)",
        'opt_input_horiz': "수평 줄무늬 (Horizontal Stripe)",
        'opt_input_flat': "단색 이미지 (Flat Image)",
        'opt_filter_vert': "수직 에지 필터 (Vertical Filter)",
        'opt_filter_horiz': "수평 에지 필터 (Horizontal Filter)",
        'btn_next': "다음 단계",
        'btn_reset': "초기화",
        'toggle_autoplay': "자동 재생",
        'input_matrix': "입력 ({r}x{c})",
        'filter_matrix': "필터 ({r}x{c})",
        'output_matrix': "출력 ({r}x{c})",
        'calc_header': "### 계산 상세 내역",
        'output_label': "\\text{출력}",
        'complete_msg': "에지 감지 스캔이 완료되었습니다!",
        'tab_matrix': "행렬 애니메이션",
        'tab_image': "실제 이미지 업로드",
        'upload_prompt': "이미지 업로드 (PNG, JPG, JPEG)",
        'original_image': "원본 이미지 (흑백)",
        'filtered_image': "에지 감지 결과",
        'opt_filter_both': "크기 (수직 + 수평)",
        'theory_title': "이론: 합성곱(Convolution)과 에지 감지",
        'theory_content': r"""### 1. 핵심 메커니즘: 합성곱 (Convolution)
CNN의 핵심 엔진은 합성곱 연산입니다. 이를 이미지 위를 구역별로 비추는 작은 손전등이라고 생각할 수 있습니다.
이 "손전등"을 **커널(Kernel)** 또는 필터(Filter)라고 부릅니다. 커널은 작은 숫자(가중치)의 행렬입니다. 일반적으로 CNN의 첫 번째 계층에서는 $3 \times 3$ 행렬을 사용합니다. 네트워크는 이 커널을 전체 입력 이미지에 걸쳐 왼쪽에서 오른쪽으로, 위에서 아래로 미끄러지듯 이동시킵니다.

각 단계에서 커널의 가중치와 현재 위치한 이미지 픽셀 값 사이에 요소별(element-wise) 곱셈을 수행합니다. 그런 다음 곱해진 모든 값을 더하여 하나의 새로운 숫자를 생성합니다. 이 과정을 통해 **특성 맵(Feature Map)**이라는 새로운 격자가 만들어집니다.

### 2. 에지(Edge)의 수학적 원리
숫자 행렬이 어떻게 실제로 윤곽선을 찾아낼까요? 이미지의 에지는 단순히 픽셀 강도가 급격하게 변하는 위치(예: 검은색에서 흰색으로의 변화)를 의미합니다.

커널 안의 특정한 숫자 패턴은 이러한 변화를 감지하는 탐지기 역할을 합니다. 대표적인 예가 **소벨 필터(Sobel Filter)**이며, CNN의 초기 계층은 자주 이 패턴을 모방합니다.

**수직 에지 필터 (Vertical Edge Filter):**
$$
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
$$
이 필터를 이미지의 단색 영역(모두 흰색이거나 모두 검은색) 위로 슬라이드하면, 왼쪽의 음수들이 오른쪽의 양수들과 완벽히 상쇄되어 합이 $0$이 됩니다. 하지만 어두운 곳에서 밝은 곳으로 전환되는 부분을 슬라이드하면, 양수 부분은 더 큰 값과 곱해지고 음수 부분은 더 작은 값과 곱해집니다. 그 결과 0이 아닌 큰 숫자가 나오며, 특성 맵에서 그 픽셀이 수학적으로 "불이 켜지면서" 수직 에지임을 나타냅니다.

**수평 에지 필터 (Horizontal Edge Filter):**
$$
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
$$
이 필터도 수직 필터와 완전히 같은 논리를 적용하지만, 위에서 아래로의 급격한 명암 변화를 감지하기 위해 회전되어 있습니다.

### 3. CNN의 구현 방식
전통적인 알고리즘은 인간이 직접 $-1$, $0$, $1$과 같은 값을 코딩해야 하지만, CNN은 다른 방식을 취합니다:
* **무작위 초기화 (Random Initialization):** CNN이 처음 생성될 때, 필터 내부의 값은 무작위의 작은 숫자로 설정됩니다. 이때 네트워크는 에지가 무엇인지 알지 못합니다.
* **순전파 (Forward Pass):** 이미지가 이 무작위 필터들을 통과하면서 무질서하고 의미 없는 특성 맵이 생성됩니다.
* **손실 계산 (Loss Calculation):** 네트워크는 예측(예: "이게 강아지인가?")을 수행하고 정답과 비교하여 오차(Loss)를 계산합니다.
* **역전파 (Backpropagation):** 네트워크는 미적분을 사용하여 오차를 줄이려면 필터 내부의 숫자를 어떻게 조정해야 하는지 알아냅니다.

수천 번의 반복을 거치면서, CNN은 에지를 감지하는 것이 형태를 이해하기 시작하는 가장 수학적으로 효율적인 방법이라는 것을 자연스럽게 깨닫습니다. 학습이 완료될 즈음 훈련된 CNN의 맨 첫 번째 계층 필터 내부를 살펴보면, 완벽하게 튜닝된 에지 탐지기인 소벨 필터와 거의 똑같은 행렬을 발견할 수 있습니다.
"""
    }
}

st.set_page_config(layout="wide", page_title=t[st.session_state.lang]['page_title'])

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
# TABS SETUP
# ==============================================================================
tab_image, tab_matrix = st.tabs([t[st.session_state.lang]['tab_image'], t[st.session_state.lang]['tab_matrix']])

# ==============================================================================
# CORE LOGIC FOR MATRIX
# ==============================================================================
def get_input_matrix(pattern_key):
    m = np.zeros((6, 6), dtype=int)
    if pattern_key == 'opt_input_vert':
        m[:, 3:] = 10
    elif pattern_key == 'opt_input_horiz':
        m[3:, :] = 10
    else: # opt_input_flat
        m[:] = 5
    return m

def get_filter_matrix(filter_key):
    if filter_key == 'opt_filter_vert':
        return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    elif filter_key == 'opt_filter_horiz':
        return np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    else: # For magnitude (won't be directly used in matrix step-by-step)
        return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

if 'ed_pattern_key' not in st.session_state:
    st.session_state.ed_pattern_key = 'opt_input_vert'
if 'ed_filter_key' not in st.session_state:
    st.session_state.ed_filter_key = 'opt_filter_vert'

if 'ed_input_matrix' not in st.session_state:
    st.session_state.ed_input_matrix = get_input_matrix(st.session_state.ed_pattern_key)
if 'ed_filter_matrix' not in st.session_state:
    st.session_state.ed_filter_matrix = get_filter_matrix(st.session_state.ed_filter_key)
if 'ed_output_matrix' not in st.session_state:
    st.session_state.ed_output_matrix = np.zeros((4, 4), dtype=int)
if 'ed_step' not in st.session_state:
    st.session_state.ed_step = 0
if 'ed_auto_play' not in st.session_state:
    st.session_state.ed_auto_play = False

def apply_settings(pattern_key, filter_key):
    st.session_state.ed_pattern_key = pattern_key
    st.session_state.ed_filter_key = filter_key
    st.session_state.ed_input_matrix = get_input_matrix(pattern_key)
    st.session_state.ed_filter_matrix = get_filter_matrix(filter_key)
    st.session_state.ed_output_matrix = np.zeros((4, 4), dtype=int)
    st.session_state.ed_step = 0
    st.session_state.ed_auto_play = False

def reset_state():
    apply_settings(st.session_state.ed_pattern_key, st.session_state.ed_filter_key)

def next_step():
    if st.session_state.ed_step < 16:
        row = st.session_state.ed_step // 4
        col = st.session_state.ed_step % 4
        
        region = st.session_state.ed_input_matrix[row:row+3, col:col+3]
        result = np.sum(region * st.session_state.ed_filter_matrix)
        st.session_state.ed_output_matrix[row, col] = result
        
        st.session_state.ed_step += 1

def get_dynamic_color(val, max_abs_val, is_input):
    if is_input:
        if max_abs_val == 0:
            return "rgba(128, 128, 128, 0.1)"
        opacity = 0.1 + 0.8 * (abs(val) / max_abs_val)
        return f"rgba(54, 162, 235, {opacity:.2f})"
    else:
        if val == 0:
            return "rgba(128, 128, 128, 0.1)"
        opacity = 0.1 + 0.8 * (abs(val) / max_abs_val) if max_abs_val != 0 else 0.5
        if val > 0:
            return f"rgba(54, 162, 235, {opacity:.2f})"
        else:
            return f"rgba(255, 75, 75, {opacity:.2f})"

def render_matrix_sequential(matrix, name, step, is_input=False, is_filter=False, is_output=False):
    html = f"<h4 style='text-align: center; margin-top: 30px;'>{name}</h4>"
    html += "<table style='border-collapse: collapse; margin: 0 auto; font-size: 1.2rem; font-family: monospace;'>"
    
    rows, cols = matrix.shape
    max_abs_val = np.max(np.abs(matrix))
    if max_abs_val == 0 and is_output and step > 0:
        max_abs_val = 30
    elif is_output:
        max_abs_val = 30
        
    current_row = step // 4 if step < 16 else -1
    current_col = step % 4 if step < 16 else -1
    
    for i in range(rows):
        html += "<tr>"
        for j in range(cols):
            val = matrix[i, j]
            color = get_dynamic_color(val, max_abs_val, is_input)
            
            border = "1px solid rgba(128, 128, 128, 0.3)"
            font_weight = "normal"
            
            if is_input and current_row != -1:
                if current_row <= i < current_row + 3 and current_col <= j < current_col + 3:
                    border = "3px solid #AB63FA"
                    font_weight = "bold"
            elif is_output:
                if current_row != -1 and i == current_row and j == current_col:
                    border = "3px solid #AB63FA"
                    font_weight = "bold"
                elif i * 4 + j >= step:
                    color = "transparent"
                    val = ""
                    border = "1px dashed rgba(128, 128, 128, 0.3)"
            elif is_filter:
                border = "1px solid rgba(128, 128, 128, 0.6)"

            html += f"<td style='width: 60px; height: 60px; text-align: center; vertical-align: middle; background-color: {color}; border: {border}; font-weight: {font_weight};'>{val}</td>"
        html += "</tr>"
    html += "</table>"
    return html

# ==============================================================================
# IMAGE UPLOAD & CONVOLUTION LOGIC
# ==============================================================================
def convolve2d_numpy(image, kernel):
    padded = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    return np.sum(windows * kernel, axis=(2, 3))

# ==============================================================================
# TAB 1: MATRIX ANIMATION
# ==============================================================================
with tab_matrix:
    st.markdown(t[st.session_state.lang]['controls_header'])

    ctrl_col1, ctrl_col2 = st.columns(2)

    with ctrl_col1:
        pattern_options = ['opt_input_vert', 'opt_input_horiz', 'opt_input_flat']
        pattern_labels = [t[st.session_state.lang][k] for k in pattern_options]
        curr_pattern_idx = pattern_options.index(st.session_state.ed_pattern_key)
        
        sel_pattern_label = st.selectbox(t[st.session_state.lang]['lbl_input'], pattern_labels, index=curr_pattern_idx)
        sel_pattern_key = pattern_options[pattern_labels.index(sel_pattern_label)]

    with ctrl_col2:
        filter_options = ['opt_filter_vert', 'opt_filter_horiz']
        filter_labels = [t[st.session_state.lang][k] for k in filter_options]
        curr_filter_idx = filter_options.index(st.session_state.ed_filter_key)
        
        sel_filter_label = st.selectbox(t[st.session_state.lang]['lbl_filter'], filter_labels, index=curr_filter_idx)
        sel_filter_key = filter_options[filter_labels.index(sel_filter_label)]

    if sel_pattern_key != st.session_state.ed_pattern_key or sel_filter_key != st.session_state.ed_filter_key:
        apply_settings(sel_pattern_key, sel_filter_key)
        st.rerun()

    action_col1, action_col2, action_col3, _ = st.columns([1, 1, 1, 5])
    with action_col1:
        if st.button(t[st.session_state.lang]['btn_next'], disabled=st.session_state.ed_step >= 16 or st.session_state.ed_auto_play, use_container_width=True):
            next_step()
            st.rerun()
    with action_col2:
        if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
            reset_state()
            st.rerun()
    with action_col3:
        auto_play = st.toggle(t[st.session_state.lang]['toggle_autoplay'], value=st.session_state.ed_auto_play)
        if auto_play != st.session_state.ed_auto_play:
            st.session_state.ed_auto_play = auto_play
            st.rerun()

    st.divider()

    in_name = t[st.session_state.lang]['input_matrix'].format(r=6, c=6)
    filt_name = t[st.session_state.lang]['filter_matrix'].format(r=3, c=3)
    out_name = t[st.session_state.lang]['output_matrix'].format(r=4, c=4)

    st.markdown(render_matrix_sequential(st.session_state.ed_input_matrix, in_name, st.session_state.ed_step, is_input=True), unsafe_allow_html=True)
    st.markdown(render_matrix_sequential(st.session_state.ed_filter_matrix, filt_name, st.session_state.ed_step, is_filter=True), unsafe_allow_html=True)
    st.markdown(render_matrix_sequential(st.session_state.ed_output_matrix, out_name, st.session_state.ed_step, is_output=True), unsafe_allow_html=True)

    st.divider()

    st.markdown(t[st.session_state.lang]['calc_header'])
    if st.session_state.ed_step < 16:
        row = st.session_state.ed_step // 4
        col = st.session_state.ed_step % 4
        
        region = st.session_state.ed_input_matrix[row:row+3, col:col+3]
        filt = st.session_state.ed_filter_matrix
        
        calc_str = ""
        terms = []
        for i in range(3):
            for j in range(3):
                terms.append(f"({region[i,j]} \\times {filt[i,j]})")
        
        calc_str = " + ".join(terms)
        st.latex(f"{t[st.session_state.lang]['output_label']}_{{{row},{col}}} = {calc_str}")
        
        values = [str(region[i,j] * filt[i,j]) for i in range(3) for j in range(3)]
        sum_str = " + ".join(values)
        result = np.sum(region * filt)
        
        st.latex(f"= {sum_str} = \\mathbf{{{result}}}")
    else:
        st.success(t[st.session_state.lang]['complete_msg'])

    if st.session_state.ed_auto_play and st.session_state.ed_step < 16:
        time.sleep(1.0)
        next_step()
        st.rerun()
    elif st.session_state.ed_auto_play and st.session_state.ed_step >= 16:
        st.session_state.ed_auto_play = False
        st.rerun()

# ==============================================================================
# TAB 2: REAL IMAGE EDGE DETECTION
# ==============================================================================
with tab_image:
    uploaded_file = st.file_uploader(t[st.session_state.lang]['upload_prompt'], type=['png', 'jpg', 'jpeg'])
    
    img_path_or_file = uploaded_file
    if uploaded_file is None:
        # default_img_path = "강아지.jpg"
        default_img_path = "DL004/강아지.jpg"
        if os.path.exists(default_img_path):
            img_path_or_file = default_img_path
            
    if img_path_or_file is not None:
        img = Image.open(img_path_or_file).convert('L')
        img.thumbnail((800, 800))
        img_arr = np.array(img, dtype=float)
        
        filter_options_img = ['opt_filter_vert', 'opt_filter_horiz', 'opt_filter_both']
        filter_labels_img = [t[st.session_state.lang][k] for k in filter_options_img]
        sel_filter_label_img = st.selectbox(t[st.session_state.lang]['lbl_filter'], filter_labels_img, key='img_filter')
        sel_filter_key_img = filter_options_img[filter_labels_img.index(sel_filter_label_img)]
        
        vert_kernel = get_filter_matrix('opt_filter_vert')
        horiz_kernel = get_filter_matrix('opt_filter_horiz')
        
        with st.spinner("Processing..."):
            if sel_filter_key_img == 'opt_filter_vert':
                edges = convolve2d_numpy(img_arr, vert_kernel)
                edges = np.abs(edges)
            elif sel_filter_key_img == 'opt_filter_horiz':
                edges = convolve2d_numpy(img_arr, horiz_kernel)
                edges = np.abs(edges)
            else: # both
                edges_v = convolve2d_numpy(img_arr, vert_kernel)
                edges_h = convolve2d_numpy(img_arr, horiz_kernel)
                edges = np.sqrt(edges_v**2 + edges_h**2)
            
            if np.max(edges) > 0:
                edges_norm = (edges / np.max(edges) * 255).astype(np.uint8)
            else:
                edges_norm = edges.astype(np.uint8)
                
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.markdown(f"**{t[st.session_state.lang]['original_image']}**")
            st.image(img, use_container_width=True)
            
        with col_img2:
            st.markdown(f"**{t[st.session_state.lang]['filtered_image']}**")
            st.image(edges_norm, use_container_width=True)
