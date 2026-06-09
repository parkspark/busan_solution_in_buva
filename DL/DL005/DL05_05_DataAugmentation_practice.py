import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import random

# ==========================================
# 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="데이터 유형별 Data Augmentation", layout="wide")

# ==========================================
# 1. 상단 화면 (제목 및 설명)
# ==========================================
st.title("데이터 유형별 Data Augmentation 시각적 탐구")
st.markdown("""
> **데이터 증강(Data Augmentation)**의 핵심 목적:
> - **데이터셋 확장**: 한정된 원본 데이터를 변형하여 학습 가능한 샘플의 수를 인위적으로 늘립니다.
> - **과적합(Overfitting) 방지**: 모델이 한정된 학습 데이터에만 지나치게 맞춰지는 현상을 방지합니다.
> - **강건성(Robustness) 향상**: 조명 변화, 노이즈, 오타 등 다양한 변수와 환경에서도 모델이 안정적으로 동작하게 만듭니다.
""")

st.markdown("---")

# ==========================================
# 탭 구성 (사이드바 대신 탭을 활용하여 뷰 분리)
# ==========================================
tab_image, tab_text = st.tabs(["🖼️ 이미지 데이터 증강", "📝 텍스트 데이터 증강"])

# ==========================================
# [탭 1] 이미지 데이터 증강 (컴퓨터 비전)
# ==========================================
with tab_image:
    st.subheader("🖼️ 이미지 데이터 증강 설정")
    
    # 1. 입력부
    uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg)", type=["jpg", "png", "jpeg"], key="img_uploader")
    
    # 2. 조작부 (메인 화면 배치, 3개의 컬럼)
    col_geom, col_color, col_region = st.columns(3)
    
    with col_geom:
        st.markdown("**1. 기하학적 변환**")
        use_h_flip = st.checkbox("좌우 반전 (Horizontal Flip)")
        use_v_flip = st.checkbox("상하 반전 (Vertical Flip)")
        rot_angle = st.slider("회전 각도", -180, 180, 0)
        scale_factor = st.slider("크기 조절 (Scale)", 0.5, 2.0, 1.0)
        
    with col_color:
        st.markdown("**2. 색상 및 픽셀 변환**")
        brightness = st.slider("밝기 조정 (Brightness)", 0.5, 2.0, 1.0)
        contrast = st.slider("대비 조정 (Contrast)", 0.5, 2.0, 1.0)
        use_noise = st.checkbox("노이즈 추가 (Gaussian Noise)")
        
    with col_region:
        st.markdown("**3. 영역 처리**")
        use_erasing = st.checkbox("Random Erasing (무작위 가림)")
        erasing_size = st.slider("가림 박스 크기 비율", 0.1, 0.5, 0.2) if use_erasing else 0.2
        
    st.markdown("---")
    
    # 3. 결과 시각화
    if uploaded_file is None:
        st.info("데이터(이미지)를 입력해주세요.")
    else:
        # 원본 이미지 로드
        original_img = Image.open(uploaded_file).convert("RGB")
        aug_img = original_img.copy()
        
        # --- 증강 파이프라인 (순차적 적용) ---
        
        # 기하학적 변환 적용
        if use_h_flip:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
        if use_v_flip:
            aug_img = aug_img.transpose(Image.FLIP_TOP_BOTTOM)
            
        if rot_angle != 0:
            aug_img = aug_img.rotate(rot_angle, expand=True)
            
        if scale_factor != 1.0:
            w, h = aug_img.size
            aug_img = aug_img.resize((int(w * scale_factor), int(h * scale_factor)))
            
        # 색상 및 픽셀 변환 적용
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(aug_img)
            aug_img = enhancer.enhance(contrast)
            
        aug_array = np.array(aug_img).astype(np.float32)
        
        if use_noise:
            # 평균 0, 표준편차 25의 가우시안 노이즈
            noise = np.random.normal(0, 25, aug_array.shape)
            aug_array = np.clip(aug_array + noise, 0, 255)
            
        # 영역 처리 (Random Erasing) 적용
        if use_erasing:
            h, w, _ = aug_array.shape
            box_h = int(h * erasing_size)
            box_w = int(w * erasing_size)
            
            # 박스가 생성될 수 있는 여유 공간이 있을 때만 적용
            if h - box_h > 0 and w - box_w > 0:
                y = np.random.randint(0, h - box_h)
                x = np.random.randint(0, w - box_w)
                # 선택된 영역을 검은색(0,0,0)으로 마스킹
                aug_array[y:y+box_h, x:x+box_w] = 0
                
        final_aug_img = aug_array.astype(np.uint8)
        
        # 2단 분할 시각화
        col_img_in, col_img_out = st.columns(2)
        with col_img_in:
            st.markdown("<h4 style='text-align: center;'>원본 이미지</h4>", unsafe_allow_html=True)
            st.image(original_img, use_container_width=True)
        with col_img_out:
            st.markdown("<h4 style='text-align: center;'>증강된 이미지</h4>", unsafe_allow_html=True)
            st.image(final_aug_img, use_container_width=True)

# ==========================================
# [탭 2] 텍스트 데이터 증강 (자연어 처리)
# ==========================================
with tab_text:
    st.subheader("📝 텍스트 데이터 증강 설정")
    
    # 1. 입력부
    default_text = "머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터로부터 학습하여 특정 작업을 수행할 수 있도록 합니다."
    user_text = st.text_area("문장을 입력하세요", value=default_text, height=100)
    
    # 2. 조작부 (메인 화면 배치)
    aug_method = st.radio(
        "증강 기법을 선택하세요",
        [
            "동의어 교체 (Random Synonym Replacement)", 
            "무작위 단어 삭제 (Random Deletion)", 
            "단어 순서 무작위 변경 (Random Swap)", 
            "번역 증강 (Back Translation - Dummy)"
        ],
        horizontal=True
    )
    
    run_text_aug = st.button("텍스트 증강 실행")
    
    st.markdown("---")
    
    # 3. 결과 시각화
    if not user_text.strip():
        st.info("데이터(텍스트)를 입력해주세요.")
    else:
        words = user_text.split()
        aug_text = ""
        
        if run_text_aug:
            # 단어 단위 변환: 동의어 교체
            if aug_method == "동의어 교체 (Random Synonym Replacement)":
                # 간단한 Rule-based 동의어 사전 (테스트용)
                synonyms = {
                    "머신러닝은": "기계학습은",
                    "인공지능의": "AI의",
                    "분야로,": "영역으로,",
                    "컴퓨터가": "기계가",
                    "데이터로부터": "정보로부터",
                    "학습하여": "배워서",
                    "작업을": "업무를",
                    "수행할": "실행할"
                }
                # 사전에 있는 단어 중 무작위 확률(50%)로 동의어 교체
                new_words = [synonyms[w] if w in synonyms and random.random() > 0.5 else w for w in words]
                aug_text = " ".join(new_words)
                
            # 단어 단위 변환: 무작위 단어 삭제
            elif aug_method == "무작위 단어 삭제 (Random Deletion)":
                if len(words) > 2:
                    # 각 단어를 20% 확률로 삭제
                    new_words = [w for w in words if random.random() > 0.2]
                    # 모든 단어가 삭제되는 것을 방지하기 위해 1개는 유지
                    if len(new_words) == 0:
                        new_words = [words[random.randint(0, len(words)-1)]]
                    aug_text = " ".join(new_words)
                else:
                    aug_text = user_text
                    
            # 문장 구조 변환: 단어 순서 무작위 변경
            elif aug_method == "단어 순서 무작위 변경 (Random Swap)":
                new_words = words.copy()
                if len(new_words) > 1:
                    # 1쌍의 단어 위치를 무작위로 교환
                    idx1, idx2 = random.sample(range(len(new_words)), 2)
                    new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
                aug_text = " ".join(new_words)
                
            # 번역 증강: Back Translation
            elif aug_method == "번역 증강 (Back Translation - Dummy)":
                # [주의] 외부 API 제약(호출 제한, 속도)을 고려하여
                # 한국어 -> 영어 -> 한국어 번역 과정을 거쳤다고 가정한 유사 더미 텍스트를 반환합니다.
                dummy_pool = [
                    "기계학습은 AI의 한 범주로서, 기계가 데이터를 통해 훈련하여 특정 임무를 해낼 수 있게 합니다.",
                    "컴퓨터 시스템이 데이터에서 학습하여 작업을 할 수 있도록 하는 것이 바로 인공지능의 한 분야인 기계학습입니다."
                ]
                aug_text = random.choice(dummy_pool)
        
        # 버튼을 누르기 전 상태 처리
        if not run_text_aug:
            aug_text = "(증강 실행 버튼을 눌러주세요)"
            
        # 2단 분할 시각화
        col_txt_in, col_txt_out = st.columns(2)
        with col_txt_in:
            st.markdown("<h4 style='text-align: center;'>원본 텍스트</h4>", unsafe_allow_html=True)
            st.info(user_text)
            
        with col_txt_out:
            st.markdown("<h4 style='text-align: center;'>증강된 텍스트</h4>", unsafe_allow_html=True)
            if run_text_aug:
                st.success(aug_text)
            else:
                st.warning(aug_text)
