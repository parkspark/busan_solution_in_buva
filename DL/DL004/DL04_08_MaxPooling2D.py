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
        'page_title': "Pooling Layers Visualizer",
        'title': "Pooling Layers: Max vs Average",
        'subtitle': "An interactive educational widget comparing the mechanisms and outcomes of MaxPooling2D and AveragePooling2D.",
        'sidebar_title': "Theory: Pooling Layers",
        'why_params_0': "### Why Params = 0?\nUnlike Convolutional or Dense layers, Pooling layers have no learnable weights (parameters). They perform a fixed, static mathematical operation (finding the maximum or calculating the average), making them highly computationally efficient and fast.",
        'translation_inv': "### Translation Invariance\nPooling helps the network become robust to small translations (shifts) in the input. For instance, if a distinct edge shifts slightly by a pixel, the maximum value might still fall within the same pooling window, yielding the exact same output. This allows the CNN to recognize features regardless of their precise location.",
        'pool_comparison': """### MaxPooling2D vs AveragePooling2D
**MaxPooling2D** extracts the most prominent feature (the maximum value) from the window.
- **Pros:** Excellent at capturing sharp, distinct features like edges or bright spots. It provides high translation invariance.
- **Cons:** Discards all other information in the window.
- **Usage:** **Heavily preferred and most commonly used** in modern CNNs for downsampling spatial dimensions.

**AveragePooling2D** calculates the mean of all values in the window.
- **Pros:** Retains a smooth, global summary of all information in the region.
- **Cons:** Tends to blur features and dilute strong, distinct signals.
- **Usage:** Rarely used for intermediate downsampling nowadays. However, **GlobalAveragePooling2D** is extremely common at the very end of a network before the final classification layer.""",
        'btn_next': "Next Step",
        'btn_reset': "Reset",
        'toggle_autoplay': "Auto Play",
        'max_pool_title': "MaxPooling2D",
        'avg_pool_title': "AveragePooling2D",
        'input_matrix': "Input Matrix (4x4)",
        'output_matrix': "Output Matrix (2x2)"
    },
    'KOR': {
        'page_title': "풀링(Pooling) 계층 시각화 도구",
        'title': "풀링 계층: Max vs Average",
        'subtitle': "MaxPooling2D와 AveragePooling2D의 동작 원리와 결과를 비교하는 대화형 교육용 위젯입니다.",
        'sidebar_title': "이론: 풀링(Pooling) 계층",
        'why_params_0': "### 왜 파라미터가 0일까요? (Why Params = 0?)\n합성곱이나 Dense 계층과 달리, 풀링 계층은 학습해야 할 가중치(파라미터)가 전혀 없습니다. 단순히 주어진 영역 내에서 최댓값이나 평균을 구하는 고정된 수학적 연산만 수행하므로 메모리와 연산 측면에서 매우 효율적입니다.",
        'translation_inv': "### 이동 불변성 (Translation Invariance)\n풀링은 네트워크가 입력 이미지의 미세한 이동(Shift)에 둔감해지도록(Robust) 도와줍니다. 특징(Feature)의 위치가 살짝 바뀌더라도 같은 풀링 윈도우 안에 있다면 동일한 최댓값을 출력하게 됩니다. 이를 통해 사물의 정확한 위치에 얽매이지 않고 패턴을 인식할 수 있습니다.",
        'pool_comparison': """### MaxPooling2D vs AveragePooling2D
**최대 풀링(MaxPooling2D)**은 영역 내에서 가장 강한 신호(최댓값)만 추출합니다.
- **장점:** 윤곽선(Edge)이나 질감 등 뚜렷하고 강한 특징을 포착하는 데 탁월하며, 이동 불변성이 뛰어납니다.
- **단점:** 최댓값 이외의 나머지 주변 정보는 모두 버려집니다.
- **사용 빈도:** **압도적으로 많이 사용됩니다.** 대부분의 현대 CNN 모델에서 중간 공간 차원을 축소할 때 사실상 표준(Standard)으로 쓰입니다.

**평균 풀링(AveragePooling2D)**은 영역 내 모든 값의 평균을 계산합니다.
- **장점:** 영역 전체의 정보를 버리지 않고 부드럽게 요약하여 보존합니다.
- **단점:** 강하고 날카로운 신호(특징)가 다른 값들과 섞여 흐려지는(Blurring) 부작용이 있습니다.
- **사용 빈도:** 중간 계층에서는 거의 사용되지 않습니다. 하지만 마지막 분류(Classification) 계층 직전에 전체 특성 맵을 1차원으로 압축하는 **전역 평균 풀링(GlobalAveragePooling2D)** 형태로는 매우 흔하게 사용됩니다.""",
        'btn_next': "다음 단계",
        'btn_reset': "초기화",
        'toggle_autoplay': "자동 재생",
        'max_pool_title': "최대 풀링 (MaxPooling2D)",
        'avg_pool_title': "평균 풀링 (AveragePooling2D)",
        'input_matrix': "입력 행렬 (4x4)",
        'output_matrix': "출력 행렬 (2x2)"
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
with st.expander(t[st.session_state.lang]['sidebar_title']):
    st.info(t[st.session_state.lang]['why_params_0'])
    st.success(t[st.session_state.lang]['translation_inv'])
    st.warning(t[st.session_state.lang]['pool_comparison'])

st.divider()

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================
if 'pool_step' not in st.session_state:
    st.session_state.pool_step = 0
if 'pool_autoplay' not in st.session_state:
    st.session_state.pool_autoplay = False

max_steps = 4 # 2x2 output

# ==============================================================================
# CONTROLS
# ==============================================================================
col_ctrl1, col_ctrl2, _ = st.columns([1.5, 1.5, 5])

with col_ctrl1:
    if st.button(t[st.session_state.lang]['btn_next'], disabled=st.session_state.pool_step >= max_steps - 1 or st.session_state.pool_autoplay, use_container_width=True):
        st.session_state.pool_step += 1
        st.rerun()
    if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
        st.session_state.pool_step = 0
        st.session_state.pool_autoplay = False
        st.rerun()

with col_ctrl2:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    auto_play = st.toggle(t[st.session_state.lang]['toggle_autoplay'], value=st.session_state.pool_autoplay)
    if auto_play != st.session_state.pool_autoplay:
        st.session_state.pool_autoplay = auto_play
        st.rerun()

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# MATRIX DATA GENERATION
# ==============================================================================
input_grid = np.array([
    [12, 20, 30,  0],
    [ 8, 12,  2,  0],
    [34, 70, 37,  4],
    [112, 100, 25, 12]
], dtype=float)

max_grid = np.zeros((2, 2), dtype=float)
avg_grid = np.zeros((2, 2), dtype=float)

step = st.session_state.pool_step

for r in range(2):
    for c in range(2):
        if r * 2 + c <= step:
            in_r = r * 2
            in_c = c * 2
            region = input_grid[in_r:in_r+2, in_c:in_c+2]
            max_grid[r, c] = np.max(region)
            avg_grid[r, c] = np.mean(region)

# ==============================================================================
# HTML RENDERING
# ==============================================================================
def render_pooling_html(matrix, name, step, is_input=False, pool_type='max'):
    html = f"<h4 style='text-align: center; margin-top: 10px;'>{name}</h4>"
    html += "<table style='border-collapse: collapse; margin: 0 auto; font-size: 1.3rem; font-family: monospace;'>"
    
    rows, cols = matrix.shape
    
    curr_r = step // 2
    curr_c = step % 2
    
    in_start_r = curr_r * 2
    in_start_c = curr_c * 2
    
    max_r, max_c = -1, -1
    if is_input and pool_type == 'max':
        region = matrix[in_start_r:in_start_r+2, in_start_c:in_start_c+2]
        idx = np.argmax(region)
        max_r = in_start_r + (idx // 2)
        max_c = in_start_c + (idx % 2)
    
    for i in range(rows):
        html += "<tr>"
        for j in range(cols):
            val = matrix[i, j]
            # Format average to 1 decimal place
            if pool_type == 'avg' and not is_input and (i * 2 + j <= step):
                str_val = f"{val:.1f}"
            else:
                str_val = f"{int(val)}"
            
            border = "1px solid currentColor"
            font_weight = "normal"
            opacity = "1.0"
            bg_color = "transparent"
            text_decor = "none"
            
            if is_input:
                if in_start_r <= i < in_start_r + 2 and in_start_c <= j < in_start_c + 2:
                    if pool_type == 'max':
                        border = "3px solid currentColor"
                        bg_color = "rgba(128, 128, 128, 0.1)"
                        if i == max_r and j == max_c:
                            font_weight = "bold"
                            text_decor = "underline"
                            bg_color = "rgba(128, 128, 128, 0.3)"
                    else:
                        border = "3px dashed currentColor"
                        bg_color = "rgba(128, 128, 128, 0.1)"
                        font_weight = "bold"
            else:
                if i == curr_r and j == curr_c:
                    border = "3px solid currentColor" if pool_type == 'max' else "3px dashed currentColor"
                    font_weight = "bold"
                    bg_color = "rgba(128, 128, 128, 0.2)"
                elif i * 2 + j > step:
                    str_val = "" 
                    border = "1px dotted currentColor"
                    opacity = "0.3"
                    
            html += f"<td style='width: 55px; height: 55px; text-align: center; vertical-align: middle; border: {border}; font-weight: {font_weight}; opacity: {opacity}; background-color: {bg_color}; text-decoration: {text_decor};'>{str_val}</td>"
            
        html += "</tr>"
    html += "</table>"
    return html

# ==============================================================================
# MAIN RENDER (SPLIT VIEW)
# ==============================================================================
col_max, col_avg = st.columns(2)

curr_r = step // 2
curr_c = step % 2
in_start_r = curr_r * 2
in_start_c = curr_c * 2
vals = input_grid[in_start_r:in_start_r+2, in_start_c:in_start_c+2].flatten()

with col_max:
    st.markdown(f"<h3 style='text-align: center;'>{t[st.session_state.lang]['max_pool_title']}</h3>", unsafe_allow_html=True)
    st.markdown(render_pooling_html(input_grid, t[st.session_state.lang]['input_matrix'], step, is_input=True, pool_type='max'), unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.info(f"**Max**({int(vals[0])}, {int(vals[1])}, {int(vals[2])}, {int(vals[3])}) = **{int(np.max(vals))}**")
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown(render_pooling_html(max_grid, t[st.session_state.lang]['output_matrix'], step, is_input=False, pool_type='max'), unsafe_allow_html=True)

with col_avg:
    st.markdown(f"<h3 style='text-align: center;'>{t[st.session_state.lang]['avg_pool_title']}</h3>", unsafe_allow_html=True)
    st.markdown(render_pooling_html(input_grid, t[st.session_state.lang]['input_matrix'], step, is_input=True, pool_type='avg'), unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.success(f"**Average**({int(vals[0])}, {int(vals[1])}, {int(vals[2])}, {int(vals[3])}) = **{np.mean(vals):.1f}**")
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown(render_pooling_html(avg_grid, t[st.session_state.lang]['output_matrix'], step, is_input=False, pool_type='avg'), unsafe_allow_html=True)

# Auto-play logic
if st.session_state.pool_autoplay and st.session_state.pool_step < max_steps - 1:
    time.sleep(0.8)
    st.session_state.pool_step += 1
    st.rerun()
elif st.session_state.pool_autoplay and st.session_state.pool_step >= max_steps - 1:
    st.session_state.pool_autoplay = False
    st.rerun()
