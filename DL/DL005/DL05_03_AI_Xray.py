import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import requests
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# ==========================================
# 1. 페이지 설정 및 CSS
# ==========================================
st.set_page_config(page_title="AI 엑스레이 카메라", layout="wide", page_icon="📸")

st.markdown("""
<style>
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0px;
}
.sub-title {
    text-align: center;
    color: #6c757d;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
</style>
<div class="main-title">📸 AI의 엑스레이 카메라</div>
<div class="sub-title">AI 모델의 얕은 층부터 깊은 층까지, 이미지를 어떻게 다르게 인식하는지 확인해보세요!</div>
""", unsafe_allow_html=True)

with st.expander("📖 알아두면 좋은 개념: 특성 맵(Feature Map) 시각화"):
    st.markdown("""
    **필터들이 입력 이미지의 어떤 부분에서 활성화(반응)했는지 직접 관찰해보는 과정입니다.**
    
    ---
    
    ### 1. 입력 데이터 차원 맞추기 (4D 텐서 규칙)
    Keras의 `Conv2D` 레이어는 **배치(Batch)** 차원이 포함된 4차원 텐서 `(batch, height, width, channels)` 형태를 입력으로 요구합니다.
    - ❌ **`train_input[0]` 사용 시:** 크기가 `(28, 28)`인 2D 배열이 추출되므로 에러가 발생합니다.
    - ✅ **`train_input[0:1]` 슬라이싱 사용 시:** 크기가 `(1, 28, 28)`로 차원이 유지되어, 이후 `(1, 28, 28, 1)` 형태로 쉽게 변환할 수 있습니다.
    
    ### 2. 특성 맵 깊이에 따른 차이
    입력 데이터가 여러 층의 Convolution을 거치면서 특성 맵의 형태가 변화합니다.
    > *예시: 입력 (1, 28, 28, 1) ⟶ 1st Conv (1, 28, 28, 32) ⟶ Pool (1, 14, 14, 32) ⟶ 2nd Conv...*

    - **얕은 층 (초기 Conv 레이어)**
      - 에지(경계선), 코너, 줄무늬 등 원본 이미지의 **디테일하고 단순한 물리적 패턴**을 포착합니다.
    - **깊은 층 (후반 Conv 레이어)**
      - 여러 단순한 패턴들이 결합되어, 사람의 눈에는 알아보기 어려운 **고차원적이고 추상적인 형태 정보**를 학습합니다. (이를 **공간 계층 구조**라고 합니다.)
    """)

# ==========================================
# 2. 모델 및 함수 정의
# ==========================================
@st.cache_resource
def load_model():
    # 분류 헤더(Top) 제외하고 VGG16 로드
    return VGG16(weights='imagenet', include_top=False)

def get_sample_image():
    """
    기본 샘플 이미지 로드. 
    로컬 경로에 고양이 사진이 없으면 Unsplash의 자동차/동물 URL을 불러옵니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base_dir, "..", "고양이.jpg")
    
    if os.path.exists(local_path):
        return Image.open(local_path).convert("RGB")
    else:
        # 웹 URL 샘플 (fallback)
        url = "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?q=80&w=800&auto=format&fit=crop"
        res = requests.get(url, timeout=10)
        return Image.open(BytesIO(res.content)).convert("RGB")

def preprocess_image(img):
    """
    4D 텐서 규칙 준수: (배치, 높이, 너비, 채널) 형태로 명시적 변환
    """
    img_resized = img.resize((224, 224))
    img_arr = np.array(img_resized)
    
    # ★ 배치 차원(Axis=0) 추가하여 (1, 224, 224, 3)의 4D 텐서로 변환 (안정성 보장)
    tensor_4d = np.expand_dims(img_arr, axis=0)
    
    return preprocess_input(tensor_4d)

def plot_feature_grid(feature_maps, num_features=16):
    """
    Feature Map(특성 맵)을 Matplotlib을 활용해 이쁜 격자(Grid) 형태로 시각화
    """
    fmap = feature_maps[0]  # 배치 차원 제거 → (H, W, C)
    total_channels = fmap.shape[-1]
    n = min(num_features, total_channels)
    
    cols = 4
    rows = math.ceil(n / cols)
    
    # 시각적으로 이쁘게 보이기 위해 다크 배경 적용
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5), facecolor='#1E1E1E')
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    
    for i, ax in enumerate(axes_flat):
        if i < n:
            # 강렬한 'plasma' 컬러맵 적용
            ax.imshow(fmap[:, :, i], cmap='plasma', aspect='auto')
            ax.set_title(f"Filter {i+1}", color='white', fontsize=10, pad=3)
        ax.axis('off')
        
    plt.tight_layout(pad=0.8)
    return fig

# ==========================================
# 3. 레이어 정보 및 인터랙티브 캡션 (깊이에 따른 매핑)
# ==========================================
LAYER_INFO = {
    1: {
        "name": "block1_conv2", 
        "title": "1단계: 얕은 층 (디테일/선/색상)",
        "desc": "🔍 지금 AI는 이미지의 외곽선, 털의 질감, 배경과의 경계선을 선명하게 뜯어보고 있습니다!"
    },
    2: {
        "name": "block2_conv2",
        "title": "2단계: 초기-중간 층 (패턴/텍스쳐)",
        "desc": "🧩 조금 더 들어왔습니다. 단순한 선들이 모여 '방향성'이나 '반복되는 패턴(텍스쳐)'으로 결합되고 있어요."
    },
    3: {
        "name": "block3_conv3",
        "title": "3단계: 중간 층 (부품/형태)",
        "desc": "💡 이제 형태가 잡히기 시작합니다. 단순한 질감을 넘어 '모서리', '원형' 같은 구체적인 부품을 인식 중입니다."
    },
    4: {
        "name": "block4_conv3",
        "title": "4단계: 중간-깊은 층 (복합 형태)",
        "desc": "🐾 눈에 보이는 형태가 많이 일그러졌죠? AI는 이제 '눈', '바퀴'와 같은 의미 있는 조각들을 조립하는 단계입니다."
    },
    5: {
        "name": "block5_conv3",
        "title": "5단계: 깊은 층 (추상적 형태/의미)",
        "desc": "🧠 깊은 층으로 오니 형체를 알아보기 힘들죠? AI는 이제 단순한 선이 아니라 '동물의 귀 모양', '바퀴의 형태' 같은 고차원적인 추상 개념을 조합하고 있습니다."
    }
}

# ==========================================
# 4. 사이드바 UI 구성
# ==========================================
with st.sidebar:
    st.header("🖼️ 이미지 업로드")
    uploaded_file = st.file_uploader("분석할 사진을 선택하세요", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    
    st.header("🎚️ AI 시선 깊이 조절")
    depth = st.slider(
        "레이어를 선택하세요",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        format="%d단계"
    )
    
    st.markdown(f"**현재 선택:**<br><span style='color:#4ECDC4;'>{LAYER_INFO[depth]['title']}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💡 **Tip:** 얕은 층은 해상도가 높고 형태가 뚜렷하지만, 깊은 층으로 갈수록 이미지가 작아지고 추상적인 패턴만 남게 됩니다.")

# ==========================================
# 5. 메인 로직 및 레이아웃 (2열 분할)
# ==========================================
# 업로드된 파일이 없으면 기본 샘플 이미지 로드
if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
else:
    pil_img = get_sample_image()

# 화면 좌우 2열 분할
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("📷 원본 이미지")
    st.image(pil_img, use_container_width=True)
    
with col_right:
    # 사용자가 슬라이더로 선택한 깊이에 맞는 정보 가져오기
    info = LAYER_INFO[depth]
    
    # 돋보기 및 인터랙티브 캡션 동적 출력
    st.success(info["desc"], icon="✨")
    
    with st.spinner("AI가 특징을 추출하고 있습니다..."):
        # 모델 로드 및 전처리
        model = load_model()
        preprocessed_img = preprocess_image(pil_img)
        
        # Functional API로 중간 레이어 추출 (가중치 공유)
        feature_model = tf.keras.Model(
            inputs=model.input, 
            outputs=model.get_layer(info["name"]).output
        )
        
        # 특성 맵 예측: 반환된 shape은 (1, H, W, C)
        feature_maps = feature_model.predict(preprocessed_img)
        
        # 시각화 관련 메타데이터 표시
        h, w, c = feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]
        st.markdown(f"**[{info['name']}] 특성 맵 크기:** `{h} x {w}` (전체 채널 {c}개 중 16개 표시)")
        
        # Grid 형태 렌더링
        fig = plot_feature_grid(feature_maps, num_features=16)
        st.pyplot(fig, use_container_width=True)
