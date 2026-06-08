import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# ==============================================================================
# SESSION STATE INITIALIZATION & LANGUAGE
# ==============================================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'KOR'

t = {
    'ENG': {
        'page_title': "Spatial Hierarchy Visualizer",
        'title': "Spatial Hierarchy in CNNs (Real Image)",
        'subtitle': "An interactive visualizer demonstrating how CNNs extract features from real images, evolving from simple edges to abstract concepts.",
        'lbl_layer': "Select CNN Layer Depth",
        'opt_shallow': "Shallow Layer (Low-level)",
        'opt_middle': "Middle Layer (Mid-level)",
        'opt_deep': "Deep Layer (High-level)",
        'btn_next': "Next Feature",
        'btn_reset': "Reset",
        'input_title': "Original Input Image",
        'feature_title': "Simulated Feature Map (Activations)",
        'uploader': "Upload an image (JPG/PNG)",
        'default_img': "Using default image."
    },
    'KOR': {
        'page_title': "공간적 계층 구조 시각화",
        'title': "CNN의 공간적 계층 구조 (실제 이미지 기반)",
        'subtitle': "실제 이미지를 입력했을 때 합성곱 신경망이 단순한 선형태부터 고도로 추상적인 개념까지 특성을 추출해내는 과정을 시각화합니다.",
        'lbl_layer': "CNN 계층 깊이 선택",
        'opt_shallow': "얕은 계층 (저수준 특성)",
        'opt_middle': "중간 계층 (중수준 특성)",
        'opt_deep': "깊은 계층 (고수준/추상적 특성)",
        'btn_next': "다음 특성 보기",
        'btn_reset': "초기화",
        'input_title': "원본 입력 이미지",
        'feature_title': "시뮬레이션된 특성 맵 (활성화 영역)",
        'uploader': "분석할 이미지 업로드 (JPG/PNG)",
        'default_img': "기본 이미지를 사용 중입니다."
    }
}

hierarchy_data = {
    'opt_shallow': {
        'desc_eng': "### Shallow Layer (Low-Level Features)\nAt this level, the CNN acts like a collection of elementary edge and color detectors. It extracts localized lines and contrast boundaries without understanding the overall shape.",
        'desc_kor': "### 얕은 계층 (저수준 특성)\n이 단계에서 CNN은 단순한 윤곽선(Edge)과 명암 감지기 역할을 합니다. 피사체의 전체적인 형태를 이해하지 못한 채, 국소적인 선과 경계선만을 추출합니다.",
        'steps': [
            {
                'name_eng': "Horizontal Edges", 'name_kor': "수평 윤곽선 (Sobel-Y)",
                'exp_eng': "Detects horizontal boundaries and structural lines across the image.",
                'exp_kor': "이미지 전체에 걸쳐 수평으로 이어진 경계선과 구조적인 선들을 감지합니다."
            },
            {
                'name_eng': "Vertical Edges", 'name_kor': "수직 윤곽선 (Sobel-X)",
                'exp_eng': "Detects vertical boundaries and structural lines across the image.",
                'exp_kor': "이미지 전체에 걸쳐 수직으로 이어진 경계선과 구조적인 선들을 감지합니다."
            },
            {
                'name_eng': "High Contrast Blobs", 'name_kor': "고대비 군집 (Blur & Threshold)",
                'exp_eng': "Detects distinct patches of high contrast, smoothing out fine details.",
                'exp_kor': "자잘한 디테일은 무시하고, 밝고 어두운 명암의 대비가 뚜렷한 큰 덩어리를 감지합니다."
            }
        ]
    },
    'opt_middle': {
        'desc_eng': "### Middle Layer (Mid-Level Features)\nHere, the CNN combines the simple lines from the shallow layer to recognize larger, more complex geometric patterns and specific structural keypoints.",
        'desc_kor': "### 중간 계층 (중수준 특성)\n이 단계에서는 얕은 계층에서 찾은 단순한 선들을 결합하여, 더 크고 복잡한 패턴이나 사물의 특정 부품(모서리, 질감 등)을 인식합니다.",
        'steps': [
            {
                'name_eng': "Corners & Keypoints", 'name_kor': "모서리 및 주요 특징점 (Harris Corners)",
                'exp_eng': "Identifies intersections of edges and structural points, simulating the detection of object parts (e.g., eyes, wheels, corners).",
                'exp_kor': "선들이 교차하는 모서리나 구조적 특징점을 찾아냅니다. (예: 동물의 눈/코, 사물의 뾰족한 부분 등)"
            },
            {
                'name_eng': "Complex Textures", 'name_kor': "복합 질감 및 윤곽 (Canny Edges)",
                'exp_eng': "Combines fine edges to detect complex boundaries and repeating textures like fur or grids.",
                'exp_kor': "얇고 세밀한 윤곽선들을 결합하여, 털의 질감이나 뚜렷한 형태적 경계 등 복잡한 패턴을 인식합니다."
            }
        ]
    },
    'opt_deep': {
        'desc_eng': "### Deep Layer (High-Level Concepts)\nThe deepest layers combine object parts into highly abstract visual concepts. The network focuses on the core semantic meaning of the image rather than raw pixels.",
        'desc_kor': "### 깊은 계층 (고수준/추상적 개념)\n가장 깊은 계층은 여러 특징들을 모아 고도로 추상화된 시각적 개념을 형성합니다. 네트워크는 픽셀이 아닌 피사체 전체의 의미(Semantic)에 집중합니다.",
        'steps': [
            {
                'name_eng': "Concept Attention Map", 'name_kor': "개념적 어텐션 맵 (Simulated Grad-CAM)",
                'exp_eng': "Highlights the core region of the image where the network's abstract 'concept' detection is focused, similar to a Class Activation Map (CAM).",
                'exp_kor': "CNN이 최종 판단을 내리기 위해 피사체의 어느 부분에 집중하고 있는지를 열화상(Heatmap) 형태로 보여줍니다. (가상의 클래스 활성화 맵 시뮬레이션)"
            }
        ]
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
# IMAGE UPLOAD & PREPROCESSING
# ==============================================================================
uploaded_file = st.file_uploader(t[st.session_state.lang]['uploader'], type=["jpg", "jpeg", "png"])

DEFAULT_IMG_PATH = "../강의자료/강아지.jpg"

if uploaded_file is not None:
    image_raw = Image.open(uploaded_file).convert('RGB')
else:
    if os.path.exists(DEFAULT_IMG_PATH):
        st.info(t[st.session_state.lang]['default_img'])
        image_raw = Image.open(DEFAULT_IMG_PATH).convert('RGB')
    else:
        # Fallback dummy image if file not found
        image_raw = Image.fromarray(np.ones((300, 300, 3), dtype=np.uint8) * 200)

image_np = np.array(image_raw)

# Resize for performance and consistent display
max_dim = 600
h, w = image_np.shape[:2]
if max(h, w) > max_dim:
    scale = max_dim / max(h, w)
    image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)))

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================
if 'layer_mode' not in st.session_state:
    st.session_state.layer_mode = 'opt_shallow'
if 'feature_step' not in st.session_state:
    st.session_state.feature_step = 0

# ==============================================================================
# CONTROLS
# ==============================================================================
layer_opts = ['opt_shallow', 'opt_middle', 'opt_deep']
layer_labels = [t[st.session_state.lang][opt] for opt in layer_opts]
curr_idx = layer_opts.index(st.session_state.layer_mode)

sel_label = st.radio(t[st.session_state.lang]['lbl_layer'], layer_labels, index=curr_idx, horizontal=True)
sel_mode = layer_opts[layer_labels.index(sel_label)]

if sel_mode != st.session_state.layer_mode:
    st.session_state.layer_mode = sel_mode
    st.session_state.feature_step = 0
    st.rerun()

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

col_ctrl1, col_ctrl2, _ = st.columns([1.5, 1.5, 7])

max_steps = len(hierarchy_data[st.session_state.layer_mode]['steps'])

with col_ctrl1:
    if st.button(t[st.session_state.lang]['btn_next'], disabled=st.session_state.feature_step >= max_steps - 1, use_container_width=True):
        st.session_state.feature_step += 1
        st.rerun()

with col_ctrl2:
    if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
        st.session_state.feature_step = 0
        st.rerun()

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

# Current Status Alert
current_data = hierarchy_data[st.session_state.layer_mode]
current_step = current_data['steps'][st.session_state.feature_step]
name_key = 'name_eng' if st.session_state.lang == 'ENG' else 'name_kor'
layer_name = t[st.session_state.lang][st.session_state.layer_mode]

status_msg = f"**Current Status:** [{layer_name}] ➔ Processing: **{current_step[name_key]}**" if st.session_state.lang == 'ENG' else f"**현재 상태:** [{layer_name}] ➔ **{current_step[name_key]}** 처리 중..."
st.info(status_msg)

st.divider()

# ==============================================================================
# OPENCV SIMULATED FEATURE EXTRACTION
# ==============================================================================
def apply_simulated_filter(img_array, mode, step):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    if mode == 'opt_shallow':
        if step == 0:
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.convertScaleAbs(sobel_y)
        elif step == 1:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            return cv2.convertScaleAbs(sobel_x)
        elif step == 2:
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
            
    elif mode == 'opt_middle':
        if step == 0:
            dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            res = np.zeros_like(gray)
            res[dst > 0.01 * dst.max()] = 255
            # Dilate to make corners visible
            res = cv2.dilate(res, np.ones((5,5), np.uint8), iterations=2)
            return res
        elif step == 1:
            edges = cv2.Canny(gray, 100, 200)
            return cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            
    elif mode == 'opt_deep':
        if step == 0:
            # Simulated pseudo-attention map (Gaussian center blob)
            h, w = gray.shape
            Y, X = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            sigma = min(h, w) / 3.0
            heatmap = np.exp(- (dist_from_center**2) / (2 * sigma**2))
            heatmap_uint8 = np.uint8(255 * heatmap)
            colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img_array, 0.4, colored_heatmap, 0.6, 0)
            return overlay
            
    return gray

feature_map = apply_simulated_filter(image_np, st.session_state.layer_mode, st.session_state.feature_step)

# ==============================================================================
# SPLIT VIEW RENDER
# ==============================================================================

desc_key = 'desc_eng' if st.session_state.lang == 'ENG' else 'desc_kor'
st.markdown(current_data[desc_key])
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

col_input, col_feature = st.columns(2)

with col_input:
    st.markdown(f"<h4 style='text-align: center;'>{t[st.session_state.lang]['input_title']}</h4>", unsafe_allow_html=True)
    st.image(image_np, use_container_width=True)

with col_feature:
    st.markdown(f"<h4 style='text-align: center;'>{t[st.session_state.lang]['feature_title']}</h4>", unsafe_allow_html=True)
    st.image(feature_map, use_container_width=True, clamp=True)
    
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

name_key = 'name_eng' if st.session_state.lang == 'ENG' else 'name_kor'
exp_key = 'exp_eng' if st.session_state.lang == 'ENG' else 'exp_kor'

st.success(f"### 🔍 {current_step[name_key]}")
st.markdown(f"**Description:** {current_step[exp_key]}")

st.progress((st.session_state.feature_step + 1) / max_steps)
