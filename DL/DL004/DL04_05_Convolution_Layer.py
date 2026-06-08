import streamlit as st
import numpy as np

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'KOR'

t = {
    'ENG': {
        'page_title': "Conv2D Parameter Calculator",
        'title': "Conv2D Parameter Calculator",
        'subtitle': "Visualize how layer parameters scale with varying input channels, kernel sizes, and filter counts.",
        'filters': "Number of Filters",
        'kernel_size': "Kernel Size (K x K)",
        'input_channels': "Input Channels",
        'formula_header': "### Parameter Calculation Formula",
        'formula_desc': "Each filter has $K \\times K \\times C_{in}$ weights plus $1$ bias term. The total parameters are this sum multiplied by the number of filters.",
        'total_params': "Total Parameters",
        'visual_header': "### Single Filter Volume Visualization",
        'visual_desc': "A single filter extends through the entire depth of the input channels."
    },
    'KOR': {
        'page_title': "Conv2D 파라미터 계산기",
        'title': "Conv2D 파라미터 계산기",
        'subtitle': "입력 채널, 커널 크기, 필터 수에 따라 파라미터가 어떻게 변하는지 시각화합니다.",
        'filters': "필터 수 (Filters)",
        'kernel_size': "커널 크기 (K x K)",
        'input_channels': "입력 채널 (Input Channels)",
        'formula_header': "### 파라미터 계산 공식",
        'formula_desc': "각 필터는 $K \\times K \\times C_{in}$ 개의 가중치와 $1$ 개의 편향(bias)을 가집니다. 총 파라미터 수는 이를 전체 필터 수에 곱한 값입니다.",
        'total_params': "총 파라미터 수",
        'visual_header': "### 단일 필터 볼륨 시각화",
        'visual_desc': "하나의 필터는 입력 채널의 전체 깊이만큼 확장됩니다."
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

st.divider()

# ==============================================================================
# FORM & CONTROLS
# ==============================================================================
col_ctrl, col_vis = st.columns([1, 2])

with col_ctrl:
    filters = st.slider(t[st.session_state.lang]['filters'], min_value=1, max_value=1024, value=32, step=1)
    kernel_size = st.slider(t[st.session_state.lang]['kernel_size'], min_value=1, max_value=11, value=3, step=2)
    input_channels = st.slider(t[st.session_state.lang]['input_channels'], min_value=1, max_value=512, value=1, step=1)

# Calculations
weights_per_filter = kernel_size * kernel_size * input_channels
params_per_filter = weights_per_filter + 1
total_params = params_per_filter * filters

with col_vis:
    st.markdown(t[st.session_state.lang]['formula_header'])
    st.markdown(t[st.session_state.lang]['formula_desc'])
    
    # Formula representation
    formula_latex = f"\\left( ( {kernel_size} \\times {kernel_size} \\times {input_channels} ) + 1 \\right) \\times {filters}"
    st.latex(f"{formula_latex} = \\mathbf{{{total_params:,}}}")
    
    # Functional highlighting for total parameters
    st.info(f"### {t[st.session_state.lang]['total_params']}: **{total_params:,}**")
    
    st.divider()
    
    st.markdown(t[st.session_state.lang]['visual_header'])
    st.markdown(t[st.session_state.lang]['visual_desc'])
    
    # Draw the SVG
    def draw_3d_filter(k, c_in):
        W = k * 15
        
        # Logarithmic scaling for depth visually looks better than strict linear scaling
        D_vis = 15 + np.log1p(c_in) * 20
        
        dx = D_vis * 0.7
        dy = D_vis * 0.5
        
        # Base coords for front face top-left
        x0 = 50
        y0 = 50 + dy
        
        color_front = "rgba(54, 162, 235, 0.8)"
        color_top = "rgba(54, 162, 235, 0.6)"
        color_side = "rgba(54, 162, 235, 0.4)"
        
        svg = '<svg width="100%" height="300" viewBox="0 0 500 300" xmlns="http://www.w3.org/2000/svg">'
        
        # Back Face
        svg += f'<rect x="{x0+dx}" y="{y0-dy}" width="{W}" height="{W}" fill="none" stroke="currentColor" stroke-opacity="0.3" />'
        
        # Top Face
        svg += f'<polygon points="{x0},{y0} {x0+W},{y0} {x0+W+dx},{y0-dy} {x0+dx},{y0-dy}" fill="{color_top}" stroke="currentColor" stroke-width="1" />'
        
        # Right Face
        svg += f'<polygon points="{x0+W},{y0} {x0+W+dx},{y0-dy} {x0+W+dx},{y0+W-dy} {x0+W},{y0+W}" fill="{color_side}" stroke="currentColor" stroke-width="1" />'
        
        # Front Face
        svg += f'<rect x="{x0}" y="{y0}" width="{W}" height="{W}" fill="{color_front}" stroke="currentColor" stroke-width="1" />'
        
        # Draw KxK grid on the front face
        for i in range(1, k):
            y_line = y0 + i * 15
            svg += f'<line x1="{x0}" y1="{y_line}" x2="{x0+W}" y2="{y_line}" stroke="currentColor" stroke-width="0.5" stroke-opacity="0.8" />'
            x_line = x0 + i * 15
            svg += f'<line x1="{x_line}" y1="{y0}" x2="{x_line}" y2="{y0+W}" stroke="currentColor" stroke-width="0.5" stroke-opacity="0.8" />'
            
        # Draw depth lines to simulate channels
        if c_in > 1:
            num_lines = min(20, c_in)
            for i in range(1, num_lines):
                frac = i / num_lines
                lx = x0 + W + dx * frac
                ly = y0 - dy * frac
                svg += f'<line x1="{lx}" y1="{ly}" x2="{lx}" y2="{ly+W}" stroke="currentColor" stroke-opacity="0.2" stroke-width="0.5" />'
                tx = x0 + dx * frac
                ty = y0 - dy * frac
                svg += f'<line x1="{tx}" y1="{ty}" x2="{lx}" y2="{ly}" stroke="currentColor" stroke-opacity="0.2" stroke-width="0.5" />'

        # Labels
        str_kernel = t[st.session_state.lang]['kernel_size'].split(' ')[0]
        str_channels = t[st.session_state.lang]['input_channels'].split(' ')[0]
        
        svg += f'<text x="{x0 + W/2}" y="{y0 + W + 25}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="currentColor" font-weight="bold">{str_kernel}: {k}x{k}</text>'
        svg += f'<text x="{x0 + W + dx + 15}" y="{y0 + W/2 - dy/2}" text-anchor="start" font-family="sans-serif" font-size="14" fill="currentColor" font-weight="bold">{str_channels}: {c_in}</text>'
        
        svg += '</svg>'
        return svg

    st.markdown(draw_3d_filter(kernel_size, input_channels), unsafe_allow_html=True)
