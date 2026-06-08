import streamlit as st
import numpy as np

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'KOR'

t = {
    'ENG': {
        'page_title': "Weight Sharing Visualizer",
        'title': "Parameter Sharing: Dense vs Conv2D",
        'subtitle': "This interactive educational visualizer directly contrasts the connection architectures of a Dense layer and a Conv2D layer to demonstrate the concept of **parameter sharing**.",
        'controls_header': "### Controls",
        'lbl_arch': "Select Architecture",
        'opt_dense': "Dense Layer Architecture",
        'opt_conv': "Conv2D Layer Architecture",
        'btn_next': "Next Calculation",
        'btn_reset': "Reset",
        'params_header': "### Total Unique Parameters Used",
        'dense_16_inputs': "16 Input Nodes",
        'dense_9_outputs': "9 Output Nodes",
        'conv_2x2_filter': "2x2 Filter (4 Params Reused)",
        'conv_3x3_output': "3x3 Output Grid",
        'conv_4x4_input': "4x4 Input Grid",
        'visualization_header': "### Visualization"
    },
    'KOR': {
        'page_title': "가중치 공유 시각화 도구",
        'title': "파라미터 공유: Dense vs Conv2D",
        'subtitle': "Dense 계층과 Conv2D 계층의 연결 구조를 직접 비교하여 **파라미터 공유(Parameter Sharing)**의 개념을 시연합니다.",
        'controls_header': "### 제어",
        'lbl_arch': "아키텍처 선택",
        'opt_dense': "Dense 계층 구조 (완전 연결)",
        'opt_conv': "Conv2D 계층 구조 (합성곱)",
        'btn_next': "다음 계산",
        'btn_reset': "초기화",
        'params_header': "### 사용된 총 고유 파라미터 수",
        'dense_16_inputs': "16개 입력 노드",
        'dense_9_outputs': "9개 출력 노드",
        'conv_2x2_filter': "2x2 필터 (파라미터 4개 재사용)",
        'conv_3x3_output': "3x3 출력 격자",
        'conv_4x4_input': "4x4 입력 격자",
        'visualization_header': "### 시각화"
    }
}

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
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

if 'ws_step' not in st.session_state:
    st.session_state.ws_step = 0
if 'ws_mode' not in st.session_state:
    st.session_state.ws_mode = 'opt_dense'

# ==============================================================================
# CONTROLS
# ==============================================================================
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    arch_options = ['opt_dense', 'opt_conv']
    arch_labels = [t[st.session_state.lang][opt] for opt in arch_options]
    curr_idx = arch_options.index(st.session_state.ws_mode)
    
    sel_label = st.radio(t[st.session_state.lang]['lbl_arch'], arch_labels, index=curr_idx)
    sel_mode = arch_options[arch_labels.index(sel_label)]

    if sel_mode != st.session_state.ws_mode:
        st.session_state.ws_mode = sel_mode
        st.session_state.ws_step = 0
        st.rerun()

with col2:
    st.markdown(t[st.session_state.lang]['controls_header'])
    if st.button(t[st.session_state.lang]['btn_next'], disabled=st.session_state.ws_step >= 8, use_container_width=True):
        st.session_state.ws_step += 1
        st.rerun()
    if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
        st.session_state.ws_step = 0
        st.rerun()

with col3:
    st.markdown(t[st.session_state.lang]['params_header'])
    if st.session_state.ws_mode == 'opt_dense':
        params = (st.session_state.ws_step + 1) * 16
        st.markdown(f"<h1 style='color: #FF4B4B;'>{params} / 144</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='color: #00CC96;'>4 / 4</h1>", unsafe_allow_html=True)
        
st.divider()

# ==============================================================================
# SVG RENDERING FUNCTIONS
# ==============================================================================
def render_dense_svg(step, text_16_inputs, text_9_outputs):
    svg = '<svg width="100%" height="450" viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg">'
    
    # Draw input nodes (flattened 16 array)
    for i in range(16):
        y = 40 + i * 24
        svg += f'<circle cx="100" cy="{y}" r="6" fill="currentColor" fill-opacity="0.5" />'
        
    # Draw output nodes (9 array)
    for j in range(9):
        y = 80 + j * 32
        opacity = "1.0" if j == step else "0.2"
        radius = "12" if j == step else "8"
        svg += f'<circle cx="700" cy="{y}" r="{radius}" fill="currentColor" fill-opacity="{opacity}" />'
    
    # Draw connections
    np.random.seed(step) # For unique visual styling at each step
    
    out_y = 80 + step * 32
    for i in range(16):
        in_y = 40 + i * 24
        stroke_dasharray = f"{np.random.randint(1, 12)}, {np.random.randint(2, 8)}"
        stroke_width = np.random.uniform(0.5, 3.5)
        opacity = np.random.uniform(0.3, 0.9)
        
        svg += f'<line x1="108" y1="{in_y}" x2="688" y2="{out_y}" stroke="currentColor" stroke-opacity="{opacity}" stroke-width="{stroke_width:.1f}" stroke-dasharray="{stroke_dasharray}" />'
        
    svg += f'<text x="60" y="20" font-family="sans-serif" font-size="16" fill="currentColor" font-weight="bold">{text_16_inputs}</text>'
    svg += f'<text x="660" y="50" font-family="sans-serif" font-size="16" fill="currentColor" font-weight="bold">{text_9_outputs}</text>'
    
    svg += '</svg>'
    return svg

def render_conv2d_svg(step, text_filter, text_out, text_in):
    svg = '''<svg width="100%" height="450" viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <pattern id="pat1" width="10" height="10" patternUnits="userSpaceOnUse">
            <rect width="10" height="10" fill="transparent" />
            <line x1="0" y1="0" x2="10" y2="10" stroke="currentColor" stroke-width="2" />
        </pattern>
        <pattern id="pat2" width="10" height="10" patternUnits="userSpaceOnUse">
            <rect width="10" height="10" fill="transparent" />
            <circle cx="5" cy="5" r="3" fill="currentColor" />
        </pattern>
        <pattern id="pat3" width="10" height="10" patternUnits="userSpaceOnUse">
            <rect width="10" height="10" fill="transparent" />
            <line x1="0" y1="5" x2="10" y2="5" stroke="currentColor" stroke-width="2" />
            <line x1="5" y1="0" x2="5" y2="10" stroke="currentColor" stroke-width="2" />
        </pattern>
        <pattern id="pat4" width="10" height="10" patternUnits="userSpaceOnUse">
            <rect width="10" height="10" fill="transparent" />
            <line x1="0" y1="10" x2="10" y2="0" stroke="currentColor" stroke-width="2" />
        </pattern>
    </defs>
    '''
    
    # 4x4 input grid (left)
    in_x_start = 50
    in_y_start = 80
    cell_size = 50
    
    out_row = step // 3
    out_col = step % 3
    
    filter_patterns = ['url(#pat1)', 'url(#pat2)', 'url(#pat3)', 'url(#pat4)']
    
    # Draw Input Grid
    for r in range(4):
        for c in range(4):
            x = in_x_start + c * cell_size
            y = in_y_start + r * cell_size
            fill = "transparent"
            stroke_width = 1
            
            # If inside the 2x2 filter window
            if out_row <= r <= out_row + 1 and out_col <= c <= out_col + 1:
                idx = (r - out_row) * 2 + (c - out_col)
                fill = filter_patterns[idx]
                stroke_width = 2
            
            svg += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="currentColor" stroke-width="{stroke_width}" />'

    # Filter separated visual (middle)
    filt_x_start = 350
    filt_y_start = 130
    for r in range(2):
        for c in range(2):
            x = filt_x_start + c * cell_size
            y = filt_y_start + r * cell_size
            idx = r * 2 + c
            fill = filter_patterns[idx]
            svg += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="currentColor" stroke-width="2" />'
            
    svg += f'<text x="{filt_x_start - 20}" y="{filt_y_start - 20}" font-family="sans-serif" font-size="16" fill="currentColor" font-weight="bold">{text_filter}</text>'

    # 3x3 output grid (right)
    out_x_start = 600
    out_y_start = 105
    for r in range(3):
        for c in range(3):
            x = out_x_start + c * cell_size
            y = out_y_start + r * cell_size
            
            is_active = (r == out_row and c == out_col)
            fill = "currentColor" if is_active else "transparent"
            opacity = "0.3" if is_active else "1.0"
            stroke_width = "3" if is_active else "1"
            
            if is_active:
                svg += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" fill-opacity="{opacity}" stroke="currentColor" stroke-width="{stroke_width}" />'
            else:
                svg += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{fill}" stroke="currentColor" stroke-width="{stroke_width}" />'
            
    svg += f'<text x="{out_x_start}" y="{out_y_start - 20}" font-family="sans-serif" font-size="16" fill="currentColor" font-weight="bold">{text_out}</text>'
    svg += f'<text x="{in_x_start}" y="{in_y_start - 20}" font-family="sans-serif" font-size="16" fill="currentColor" font-weight="bold">{text_in}</text>'

    # Draw connection from active filter to output cell
    active_out_x = out_x_start + out_col * cell_size + cell_size / 2
    active_out_y = out_y_start + out_row * cell_size + cell_size / 2
    
    filt_center_x = filt_x_start + cell_size
    filt_center_y = filt_y_start + cell_size
    
    svg += f'<line x1="{filt_center_x}" y1="{filt_center_y}" x2="{active_out_x}" y2="{active_out_y}" stroke="currentColor" stroke-width="3" stroke-dasharray="8,4" stroke-opacity="0.6" />'

    svg += '</svg>'
    return svg

# ==============================================================================
# MAIN RENDER
# ==============================================================================
st.markdown(t[st.session_state.lang]['visualization_header'])
if st.session_state.ws_mode == 'opt_dense':
    st.markdown(render_dense_svg(st.session_state.ws_step, t[st.session_state.lang]['dense_16_inputs'], t[st.session_state.lang]['dense_9_outputs']), unsafe_allow_html=True)
else:
    st.markdown(render_conv2d_svg(st.session_state.ws_step, t[st.session_state.lang]['conv_2x2_filter'], t[st.session_state.lang]['conv_3x3_output'], t[st.session_state.lang]['conv_4x4_input']), unsafe_allow_html=True)
