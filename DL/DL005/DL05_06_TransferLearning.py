import streamlit as st
import numpy as np
from PIL import Image, ImageFilter
import requests
from io import BytesIO

# ==========================================
# 페이지 기본 설정
# ==========================================
st.set_page_config(page_title="Transfer Learning Simulator", layout="wide", page_icon="🧠")

# ==========================================
# 1. 타이틀 및 인트로
# ==========================================
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🧠 지식 보존 vs 파괴 시뮬레이터</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-9align: center; color: #7F8C8D;'>사전 학습된 MobileNetV2의 가중치를 활용하여 전이 학습과 미세 조정을 진행합니다.<br>학습률과 동결 여부가 AI의 기존 지식을 어떻게 보존하거나 파괴하는지 직접 실험해보세요!</h4>", unsafe_allow_html=True)

with st.expander("📖 알아두면 좋은 개념: 미세 조정(Fine-Tuning)과 파괴적 망각"):
    st.markdown("""
    **미세 조정(Fine-Tuning)**은 전이 학습 이후 성능을 한 단계 더 끌어올리기 위해, 사전 학습된 모델의 일부 상위 가중치 층을 동결 해제하고 우리의 새로운 데이터와 함께 미세하게 업데이트하는 기법입니다.
    
    ---
    
    ### ⚠️ 높은 LR(학습률)의 위험성: 파괴적 망각 (Catastrophic Forgetting)
    - ❌ **높은 LR 사용 시:** ImageNet 1,400만 장을 학습하며 견고하게 쌓아 올린 사전 가중치가, 높은 학습률의 거친 경사하강법(Gradient Descent)으로 인해 한순간에 망가지고 오염됩니다.
    - ✅ **낮은 LR 사용 (대략 1e-4 이하):** 기존의 훌륭한 시각적 지식 구조를 최대한 훼손하지 않으면서, 새로운 태스크에 부드럽게 조화되도록 가중치 궤적을 미세하게 조정합니다.
    """)

st.markdown("---")

# ==========================================
# 2. [섹션 1] 실험실 제어판 (Main Layout)
# ==========================================
st.header("🎛️ 실험실 제어판")

# 사이드바 없이 메인 레이아웃에 2열로 배치
col_ctrl, col_status = st.columns(2)

with col_ctrl:
    st.subheader("1. 학습 설정")
    
    # 학습 단계 선택 (동결 여부)
    phase = st.radio(
        "학습 단계 선택 (모델 동결 여부)",
        (
            "1단계: 전이 학습 (Base Model 동결, trainable=False)",
            "2단계: 미세 조정 (상위 30개 레이어 해제, trainable=True)"
        )
    )
    
    # 학습률 선택
    lr_label = st.selectbox(
        "학습률 (Learning Rate) 선택",
        (
            "1e-2 (위험! 매우 높음)",
            "1e-3 (높음)",
            "1e-4 (권장: 미세 조정 최적)",
            "1e-5 (매우 낮음)"
        )
    )
    
    run_sim = st.button("🚀 시뮬레이션 실행", use_container_width=True)

with col_status:
    st.subheader("2. 상태 모니터")
    
    # 상태 판별 및 파괴도 연산 로직
    is_phase_2 = "2단계" in phase
    
    # 초기 상태 및 모니터링 출력
    if is_phase_2:
        if "1e-2" in lr_label:
            st.error("🚨 **경고: 파괴적 망각(Catastrophic Forgetting) 발생!**\n\n너무 높은 학습률로 인해 1,400만 장의 ImageNet 데이터로 학습된 기존의 훌륭한 지식이 산산조각 나고 있습니다.")
            contamination = 95
            status_text = "파괴적 망각 진행 중..."
        elif "1e-3" in lr_label:
            st.warning("⚠️ **주의: 지식 손실 가능성**\n\n학습률이 다소 높습니다. 기존 가중치의 미세한 윤곽 정보가 일부 훼손될 수 있습니다.")
            contamination = 40
            status_text = "일부 지식 훼손 발생"
        elif "1e-4" in lr_label:
            st.success("✨ **안정적: 완벽한 미세 조정(Fine-Tuning)**\n\n기존의 훌륭한 시각적 지식 구조를 훼손하지 않으면서 새로운 태스크에 부드럽게 조화되고 있습니다.")
            contamination = 5
            status_text = "안정적인 지식 융합 (최적)"
        else: # 1e-5
            st.info("🐢 **안내: 학습 속도 저하**\n\n가중치는 온전히 보존되지만, 학습률이 너무 낮아 새로운 태스크를 학습하는 데 지나치게 오랜 시간이 걸립니다.")
            contamination = 1
            status_text = "지식 보존 (매우 느린 학습)"
    else:
        # 1단계 (동결 상태)
        st.info("🔒 **Base Model 동결 상태 (Frozen)**\n\n사전 학습된 모델의 가중치가 잠겨 있으므로, 학습률이 아무리 높아도 기존의 시각적 지식은 절대 파괴되지 않습니다.\n\n*(단, 부착된 새로운 분류기 가중치는 요동칠 수 있습니다.)*")
        contamination = 0
        status_text = "지식 절대 보존 (가중치 잠금)"

st.markdown("---")

# ==========================================
# 3. [섹션 2] 실시간 지식 파괴도 & 특성 맵 시각화
# ==========================================
st.header("🔬 실시간 지식 파괴도 및 특성 맵 시각화")

# 더미 샘플 이미지 로드 함수 (캐싱하여 반복 로딩 방지)
@st.cache_data
def load_sample_image():
    # 고양이 샘플 이미지 URL (Unsplash)
    url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?q=80&w=800&auto=format&fit=crop"
    try:
        res = requests.get(url, timeout=5)
        img = Image.open(BytesIO(res.content)).convert("RGB")
        return img
    except:
        # 통신 실패 시 회색 배경의 더미 이미지 반환
        return Image.new('RGB', (400, 400), color='gray')

# 사용자 이미지 업로드 추가
uploaded_file = st.file_uploader("자신만의 이미지를 업로드하여 테스트해 보세요! (선택 사항)", type=["png", "jpg", "jpeg"])

if run_sim:
    # 3-1. 가중치 오염도 게이지 (Progress Bar)
    st.markdown(f"**가중치 오염도 (Catastrophic Forgetting 지수): {contamination}%** - *{status_text}*")
    progress_bar = st.progress(0)
    progress_bar.progress(contamination / 100.0)
    
    # 3-2. 특성 맵 시각화 (좌우 배치)
    if uploaded_file is not None:
        orig_img = Image.open(uploaded_file).convert("RGB")
    else:
        orig_img = load_sample_image()
    
    col_img_in, col_img_out = st.columns(2)
    
    with col_img_in:
        st.markdown("<h4 style='text-align: center;'>원본 이미지</h4>", unsafe_allow_html=True)
        st.image(orig_img, use_container_width=True)
        
    with col_img_out:
        st.markdown("<h4 style='text-align: center;'>AI가 인식하는 특성 맵 (Feature Map)</h4>", unsafe_allow_html=True)
        
        # 오염도에 따른 특성 맵(필터 효과) 더미 처리
        if contamination >= 90:
            # 1e-2 (파괴적 망각): 형체를 알아볼 수 없는 극심한 노이즈
            img_arr = np.array(orig_img.convert("L")) # 흑백 변환
            noise = np.random.normal(128, 100, img_arr.shape) # 강한 정규분포 노이즈
            noisy_img = np.clip(noise, 0, 255).astype(np.uint8)
            out_img = Image.fromarray(noisy_img)
            caption = "💥 지식 완전 파괴: 의미 있는 패턴 추출 실패"
            
        elif contamination >= 40:
            # 1e-3 (부분 훼손): 윤곽선이 뭉개지고 노이즈가 발생
            blurred = orig_img.convert("L").filter(ImageFilter.GaussianBlur(radius=5))
            img_arr = np.array(blurred)
            noise = np.random.normal(0, 30, img_arr.shape)
            noisy_img = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            out_img = Image.fromarray(noisy_img)
            caption = "⚠️ 지식 부분 훼손: 윤곽선 흐릿함 및 형태 손실"
            
        else:
            # 1e-4 이하 또는 동결 (정상): 뚜렷한 특징선(Edge) 추출
            # 에지 검출 필터로 특징 맵 시각화
            out_img = orig_img.convert("L").filter(ImageFilter.FIND_EDGES)
            caption = "✨ 지식 보존: 뚜렷한 외곽선 및 질감 추출 성공"
            
        st.image(out_img, use_container_width=True, caption=caption)

else:
    st.info("👆 위에서 **[🚀 시뮬레이션 실행]** 버튼을 눌러 시각화 결과를 확인하세요.")

st.markdown("---")

# ==========================================
# 4. [섹션 3] 대화형 Keras 코드 룸
# ==========================================
st.header("💻 대화형 Keras 코드 룸")
st.markdown("사용자가 선택한 **학습 단계**와 **학습률**이 코드에 어떻게 반영되는지 실시간으로 확인하세요.")

# 학습률 수치만 추출 (예: "1e-4 (권장...)" -> "1e-4")
lr_value = lr_label.split(" ")[0]

# 학습 단계에 따른 동결 해제 코드 변경
if is_phase_2:
    freeze_code = """# [2단계] 미세 조정 (Fine-Tuning)
# Base Model의 가중치 잠금을 풀고, 상위 30개 레이어만 세밀하게 업데이트합니다.
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False"""
else:
    freeze_code = """# [1단계] 전이 학습 (Transfer Learning)
# 사전 학습된 가중치를 보존하기 위해 Base Model 전체를 동결(Freeze)합니다.
base_model.trainable = False"""

# 동적 Keras 코드 템플릿 작성
code_template = f"""from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# 1. 사전 학습된 모델 로드 (가중치 포함, 분류 헤더 제외)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

{freeze_code}

# 2. 커스텀 분류기(Head) 부착
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax') # 10개 클래스 분류 예시
])

# 3. 모델 컴파일 (설정된 학습률 적용)
# 현재 학습률: {lr_value}
optimizer = Adam(learning_rate={lr_value})
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 모델 학습
# model.fit(train_dataset, epochs=10, validation_data=val_dataset)
"""

# 코드 출력
st.code(code_template, language="python")
