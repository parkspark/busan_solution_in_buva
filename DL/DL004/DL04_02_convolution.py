import streamlit as st
import numpy as np
import time

# ==============================================================================
# INITIALIZE SESSION STATE (For Language and State persistence)
# ==============================================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'KOR'

if 'in_rows' not in st.session_state:
    st.session_state.in_rows = 6
if 'in_cols' not in st.session_state:
    st.session_state.in_cols = 6
if 'f_rows' not in st.session_state:
    st.session_state.f_rows = 3
if 'f_cols' not in st.session_state:
    st.session_state.f_cols = 3

def init_matrices():
    st.session_state.input_matrix = np.random.randint(0, 10, (st.session_state.in_rows, st.session_state.in_cols))
    st.session_state.filter_matrix = np.random.randint(-2, 3, (st.session_state.f_rows, st.session_state.f_cols))
    out_r = st.session_state.in_rows - st.session_state.f_rows + 1
    out_c = st.session_state.in_cols - st.session_state.f_cols + 1
    st.session_state.output_matrix = np.zeros((out_r, out_c), dtype=int)
    st.session_state.step = 0
    st.session_state.auto_play = False

if 'input_matrix' not in st.session_state:
    init_matrices()

# ==============================================================================
# TRANSLATIONS DICTIONARY (Bilingual Support)
# ==============================================================================
t = {
    'ENG': {
        'page_title': "2D Convolution Visualizer",
        'title': "2D Convolution Visualizer",
        'subtitle': "This interactive visualizer demonstrates the process of 2D convolution. A filter slides over an input matrix to produce an output matrix.",
        'controls_header': "### Controls",
        'btn_next': "Next Step",
        'btn_reset': "Reset Matrices",
        'toggle_autoplay': "Auto Play",
        'input_matrix': "Input ({r}x{c})",
        'filter_matrix': "Filter ({r}x{c})",
        'output_matrix': "Output ({r}x{c})",
        'calc_header': "### Calculation Details",
        'output_label': "\\text{Output}",
        'complete_msg': "Convolution complete! Click **Reset Matrices** to generate new matrices or change settings in the sidebar.",
        'settings': "⚙️ Settings",
        'input_size': "Input Matrix Size (Max 12x12)",
        'filter_size': "Filter Size",
        'rows': "Rows",
        'cols': "Columns",
        'err_filter_size': "Filter size cannot be larger than input size!"
    },
    'KOR': {
        'page_title': "2D 합성곱",
        'title': "2D 합성곱",
        'subtitle': "2D 합성곱 과정. 필터(커널)가 입력 행렬 위를 슬라이딩하며 출력 행렬을 생성",
        'controls_header': "### 제어",
        'btn_next': "다음 단계",
        'btn_reset': "행렬 초기화",
        'toggle_autoplay': "자동 재생",
        'input_matrix': "입력 ({r}x{c})",
        'filter_matrix': "필터 ({r}x{c})",
        'output_matrix': "출력 ({r}x{c})",
        'calc_header': "### 계산 상세 내역",
        'output_label': "\\text{출력}",
        'complete_msg': "합성곱이 완료되었습니다! **행렬 초기화**를 클릭하여 새로운 행렬을 생성하거나 사이드바에서 설정을 변경하세요.",
        'settings': "⚙️ 설정",
        'input_size': "입력 행렬 크기 (최대 12x12)",
        'filter_size': "필터 크기",
        'rows': "행 (Rows)",
        'cols': "열 (Columns)",
        'err_filter_size': "필터 크기는 입력 크기보다 클 수 없습니다!"
    }
}

st.set_page_config(layout="wide", page_title=t[st.session_state.lang]['page_title'])

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.header(t[st.session_state.lang]['settings'])
    
    st.subheader(t[st.session_state.lang]['input_size'])
    new_in_rows = st.slider(t[st.session_state.lang]['rows'], 3, 12, st.session_state.in_rows, key="sl_in_r")
    new_in_cols = st.slider(t[st.session_state.lang]['cols'], 3, 12, st.session_state.in_cols, key="sl_in_c")
    
    st.subheader(t[st.session_state.lang]['filter_size'])
    new_f_rows = st.slider(t[st.session_state.lang]['rows'] + " ", 2, 7, st.session_state.f_rows, key="sl_f_r")
    new_f_cols = st.slider(t[st.session_state.lang]['cols'] + " ", 2, 7, st.session_state.f_cols, key="sl_f_c")

    if new_f_rows > new_in_rows or new_f_cols > new_in_cols:
        st.error(t[st.session_state.lang]['err_filter_size'])
    else:
        if (new_in_rows != st.session_state.in_rows or 
            new_in_cols != st.session_state.in_cols or
            new_f_rows != st.session_state.f_rows or
            new_f_cols != st.session_state.f_cols):
            st.session_state.in_rows = new_in_rows
            st.session_state.in_cols = new_in_cols
            st.session_state.f_rows = new_f_rows
            st.session_state.f_cols = new_f_cols
            init_matrices()
            st.rerun()

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
        if st.button("ENG", type="primary" if st.session_state.lang == 'ENG' else "secondary", use_container_width=True):
            st.session_state.lang = 'ENG'
            st.rerun()
    with lang_col2:
        if st.button("KOR", type="primary" if st.session_state.lang == 'KOR' else "secondary", use_container_width=True):
            st.session_state.lang = 'KOR'
            st.rerun()

def reset():
    init_matrices()

# Dynamically calculate layout properties based on inputs
out_r = st.session_state.in_rows - st.session_state.f_rows + 1
out_c = st.session_state.in_cols - st.session_state.f_cols + 1
total_steps = out_r * out_c

def next_step():
    if st.session_state.step < total_steps:
        # Calculate the current cell
        row = st.session_state.step // out_c
        col = st.session_state.step % out_c
        
        region = st.session_state.input_matrix[row:row+st.session_state.f_rows, col:col+st.session_state.f_cols]
        result = np.sum(region * st.session_state.filter_matrix)
        st.session_state.output_matrix[row, col] = result
        
        st.session_state.step += 1

# ==============================================================================
# CONTROLS
# ==============================================================================
st.markdown(t[st.session_state.lang]['controls_header'])
col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
with col1:
    if st.button(t[st.session_state.lang]['btn_next'], disabled=st.session_state.step >= total_steps or st.session_state.auto_play, use_container_width=True):
        next_step()
        st.rerun()
with col2:
    if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
        reset()
        st.rerun()
with col3:
    auto_play = st.toggle(t[st.session_state.lang]['toggle_autoplay'], value=st.session_state.auto_play)
    if auto_play != st.session_state.auto_play:
        st.session_state.auto_play = auto_play
        st.rerun()

st.divider()

# ==============================================================================
# STYLING & RENDERING
# ==============================================================================
def get_heatmap_color(val, min_val, max_val, color_rgb):
    if max_val == min_val:
        opacity = 0.5
    else:
        # Scale between 0.1 and 0.8 to keep text readable
        opacity = 0.1 + 0.7 * ((val - min_val) / (max_val - min_val))
    return f"rgba({color_rgb}, {opacity:.2f})"

def render_matrix(matrix, name, step, is_input=False, is_filter=False, is_output=False):
    html = f"<h4 style='text-align: center;'>{name}</h4>"
    html += "<table style='border-collapse: collapse; margin: 0 auto; font-size: 1.1rem; font-family: monospace;'>"
    
    rows, cols = matrix.shape
    
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    current_row = step // out_c if step < total_steps else -1
    current_col = step % out_c if step < total_steps else -1
    
    # Colors (R, G, B)
    if is_input:
        color_rgb = "54, 162, 235"  # Blue
    elif is_filter:
        color_rgb = "75, 192, 192"  # Teal
    else:
        color_rgb = "153, 102, 255" # Purple
        
    for i in range(rows):
        html += "<tr>"
        for j in range(cols):
            val = matrix[i, j]
            color = get_heatmap_color(val, min_val, max_val, color_rgb)
            
            # Default styling
            border = "1px solid rgba(128, 128, 128, 0.3)"
            font_weight = "normal"
            
            # Highlighting logic
            if is_input and current_row != -1:
                if current_row <= i < current_row + st.session_state.f_rows and current_col <= j < current_col + st.session_state.f_cols:
                    border = "3px solid #ff4b4b"
                    font_weight = "bold"
            elif is_output:
                if current_row != -1 and i == current_row and j == current_col:
                    border = "3px solid #ff4b4b"
                    font_weight = "bold"
                elif i * out_c + j >= step:
                    # Not computed yet
                    color = "transparent"
                    val = ""
                    border = "1px dashed rgba(128, 128, 128, 0.3)"
            elif is_filter:
                border = "1px solid rgba(128, 128, 128, 0.6)"

            html += f"<td style='width: 45px; height: 45px; text-align: center; vertical-align: middle; background-color: {color}; border: {border}; font-weight: {font_weight};'>{val}</td>"
        html += "</tr>"
    html += "</table>"
    return html

# Render matrices
col_in, col_filt, col_out = st.columns(3)

in_name = t[st.session_state.lang]['input_matrix'].format(r=st.session_state.in_rows, c=st.session_state.in_cols)
filt_name = t[st.session_state.lang]['filter_matrix'].format(r=st.session_state.f_rows, c=st.session_state.f_cols)
out_name = t[st.session_state.lang]['output_matrix'].format(r=out_r, c=out_c)

with col_in:
    st.markdown(render_matrix(st.session_state.input_matrix, in_name, st.session_state.step, is_input=True), unsafe_allow_html=True)

with col_filt:
    st.markdown(render_matrix(st.session_state.filter_matrix, filt_name, st.session_state.step, is_filter=True), unsafe_allow_html=True)

with col_out:
    st.markdown(render_matrix(st.session_state.output_matrix, out_name, st.session_state.step, is_output=True), unsafe_allow_html=True)

st.divider()

# ==============================================================================
# CALCULATION DETAILS
# ==============================================================================
st.markdown(t[st.session_state.lang]['calc_header'])
if out_r <= 0 or out_c <= 0:
    st.error(t[st.session_state.lang]['err_filter_size'])
elif st.session_state.step < total_steps:
    row = st.session_state.step // out_c
    col = st.session_state.step % out_c
    
    region = st.session_state.input_matrix[row:row+st.session_state.f_rows, col:col+st.session_state.f_cols]
    filt = st.session_state.filter_matrix
    
    # If the filter is too large, the math formula overflows the screen width
    if st.session_state.f_rows * st.session_state.f_cols <= 9:
        calc_str = ""
        terms = []
        for i in range(st.session_state.f_rows):
            for j in range(st.session_state.f_cols):
                terms.append(f"({region[i,j]} \\times {filt[i,j]})")
        
        calc_str = " + ".join(terms)
        st.latex(f"{t[st.session_state.lang]['output_label']}_{{{row},{col}}} = {calc_str}")
        
        values = [str(region[i,j] * filt[i,j]) for i in range(st.session_state.f_rows) for j in range(st.session_state.f_cols)]
        sum_str = " + ".join(values)
        result = np.sum(region * filt)
        
        st.latex(f"= {sum_str} = \\mathbf{{{result}}}")
    else:
        # Simplified formula for large filters
        result = np.sum(region * filt)
        st.latex(f"{t[st.session_state.lang]['output_label']}_{{{row},{col}}} = \\sum (\\text{{Region}} \\odot \\text{{Filter}}) = \\mathbf{{{result}}}")
        
else:
    st.success(t[st.session_state.lang]['complete_msg'])

# Auto-play logic
if st.session_state.auto_play and st.session_state.step < total_steps:
    time.sleep(1.0)  # Wait 1 second before calculating next step so user can observe
    next_step()
    st.rerun()
elif st.session_state.auto_play and st.session_state.step >= total_steps:
    st.session_state.auto_play = False
    st.rerun()
