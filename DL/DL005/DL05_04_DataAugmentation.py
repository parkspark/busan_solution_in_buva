import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2

# ==========================================
# 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="Dropout vs Data Augmentation", layout="wide")

# ==========================================
# 1. 상단 화면 (제목 및 설명)
# ==========================================
st.title("Dropout vs 데이터 증강 (Data Augmentation) 시각적 비교")
st.markdown("""
> **데이터 증강**이란 기존 학습 데이터에 인위적인 변형이나 조작을 가해 데이터셋의 크기와 다양성을 인위적으로 확장하는 기법입니다. 
> 머신러닝 모델의 과적합(Overfitting)을 방지하고 일반화(Generalization) 성능을 높이는 데 필수적으로 사용됩니다.
""")

st.markdown("---")

# ==========================================
# 2. 조작부 (Controls - 메인 화면 배치)
# ==========================================
st.subheader("⚙️ 설정 패널")

# 이미지 업로드 컴포넌트
uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

# 설정 패널 좌우 배치 (사이드바 대신 메인 화면에 배치)
col_aug_set, col_drop_set = st.columns(2)

with col_aug_set:
    st.markdown("### 🎨 Data Augmentation 설정")
    # 증강 기법 선택 및 강도 조절
    
    use_flip = st.checkbox("좌우 반전 (Horizontal Flip)")
    
    use_rotation = st.checkbox("회전 (Rotation)")
    rot_angle = st.slider("회전 각도", -180, 180, 0) if use_rotation else 0
    
    use_brightness = st.checkbox("밝기 조절 (Color Jitter / Brightness)")
    brightness_factor = st.slider("밝기 강도", 0.1, 3.0, 1.0) if use_brightness else 1.0
    
    use_noise = st.checkbox("노이즈 추가 (Gaussian Noise)")
    noise_intensity = st.slider("노이즈 강도", 0.0, 1.0, 0.1) if use_noise else 0.0

with col_drop_set:
    st.markdown("### 🕳️ Dropout 설정")
    # Dropout 비율 조절 (0.0 ~ 0.9)
    dropout_rate = st.slider("Dropout 비율", 0.0, 0.9, 0.0, 0.1)

st.markdown("---")

# ==========================================
# 3. 결과 시각화 화면 (Visualizations)
# ==========================================
if uploaded_file is None:
    # 이미지가 없을 경우 예외 처리
    st.info("이미지를 업로드해주세요.")
else:
    # 원본 이미지 로드
    original_img = Image.open(uploaded_file).convert("RGB")
    original_array = np.array(original_img)
    
    # 원본 이미지 시각화 (중앙 정렬 느낌으로 배치)
    st.markdown("<h3 style='text-align: center;'>원본 이미지</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(original_img, use_container_width=True)
        
    st.markdown("---")
    
    # 결과 시각화 2단 컬럼
    col_aug_res, col_drop_res = st.columns(2)
    
    # ----------------------------------------
    # Data Augmentation 적용 처리
    # ----------------------------------------
    aug_img = original_img.copy()
    
    # 여러 기법을 순차적으로 중첩 적용
    
    # 1) 좌우 반전
    if use_flip:
        aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        
    # 2) 회전 (expand=True로 설정하여 회전 시 이미지가 잘리지 않게 함)
    if use_rotation:
        aug_img = aug_img.rotate(rot_angle, expand=True)
        
    # 3) 밝기 조절
    if use_brightness:
        enhancer = ImageEnhance.Brightness(aug_img)
        aug_img = enhancer.enhance(brightness_factor)
        
    # 4) 노이즈 추가
    aug_array = np.array(aug_img).astype(np.float32)
    if use_noise and noise_intensity > 0:
        # 픽셀별 무작위 노이즈 생성 (평균 0, 표준편차 = 강도 * 255)
        noise = np.random.normal(0, noise_intensity * 255, aug_array.shape)
        aug_array = np.clip(aug_array + noise, 0, 255)
    
    final_aug_img = aug_array.astype(np.uint8)
    
    with col_aug_res:
        st.markdown("<h3 style='text-align: center;'>Data Augmentation 적용 결과</h3>", unsafe_allow_html=True)
        st.image(final_aug_img, use_container_width=True)
        
    # ----------------------------------------
    # Dropout 적용 처리
    # ----------------------------------------
    drop_array = original_array.copy().astype(np.float32)
    
    if dropout_rate > 0:
        # 픽셀 레벨의 Dropout 구현 (RGB 채널 모두 한 번에 동일하게 차단하기 위해 2D 마스크 생성)
        # dropout_rate 확률로 0, 그 외에는 1을 갖는 이항 분포 마스크 생성
        mask = np.random.binomial(1, 1 - dropout_rate, drop_array.shape[:2])
        mask = np.expand_dims(mask, axis=-1)  # (H, W, 1) 형태로 브로드캐스팅 가능하게 변환
        
        # 원본 배열에 마스크를 곱하여 픽셀을 0(검은색)으로 만듦
        drop_array = drop_array * mask
        
    final_drop_img = drop_array.astype(np.uint8)
    
    with col_drop_res:
        st.markdown("<h3 style='text-align: center;'>Dropout 적용 결과</h3>", unsafe_allow_html=True)
        st.image(final_drop_img, use_container_width=True)

st.markdown("---")

# ==========================================
# 4. 핵심 차이 요약
# ==========================================
st.markdown("### 📊 핵심 차이 요약")
st.markdown("""
| 구분 | Dropout | 데이터 증강 |
| :--- | :--- | :--- |
| 작동 위치 | 모델 내부 (네트워크) | 모델 외부 (입력 데이터) |
| 동작 원리 | 내부 노드를 무작위로 차단 | 입력 이미지/텍스트 형태를 다양하게 왜곡 |
| 주요 효과 | 모델의 유효 복잡도 감소 | 데이터의 표현적 다양성 극대화 |
""")
