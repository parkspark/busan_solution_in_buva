# =============================================================================
# app.py
# 주제: Sequential vs Functional API 인터랙티브 이미지 시각화 교육 앱
# 레이아웃: 좌측 = Sequential 모델 결과 / 우측 = Functional API Feature Map
# 실행: uv run streamlit run app.py
# =============================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import math

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

# =============================================================================
# ① 페이지 기본 설정
# =============================================================================
st.set_page_config(
    page_title="Sequential vs Functional API",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# ② 커스텀 CSS
# =============================================================================
st.markdown("""
<style>
/* ── DL05_01_ConvFilter.py 와 동일한 톤앤매너 ── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif;
}

/* ── 타이틀 ── */
.main-title {
    text-align: center;
    padding: 1.2rem 0 0.4rem 0;
}
.main-title h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4F46E5 0%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.main-title p {
    font-size: 1rem;
    color: var(--text-color);
    opacity: 0.7;
}

/* ── 섹션 카드 ── */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
.glass-card:hover {
    box-shadow: 0 8px 25px rgba(79,70,229,0.08);
    border-color: rgba(79,70,229,0.35);
}

/* ── 좌측 패널 헤더 (Sequential) ── */
.panel-header-seq {
    background: rgba(79, 70, 229, 0.06);
    border-left: 5px solid #4F46E5;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
}
.panel-header-seq h2 {
    color: #4F46E5;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
}
.panel-header-seq p {
    color: var(--text-color);
    opacity: 0.6;
    font-size: 0.82rem;
    margin: 0.2rem 0 0;
}

/* ── 우측 패널 헤더 (Functional API) ── */
.panel-header-func {
    background: rgba(236, 72, 153, 0.06);
    border-left: 5px solid #EC4899;
    border-radius: 0 12px 12px 0;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
}
.panel-header-func h2 {
    color: #EC4899;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
}
.panel-header-func p {
    color: var(--text-color);
    opacity: 0.6;
    font-size: 0.82rem;
    margin: 0.2rem 0 0;
}

/* ── 경고 박스 (Sequential 한계) ── */
.warn-box {
    background: rgba(239, 68, 68, 0.06);
    border-left: 4px solid #EF4444;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text-color);
}

/* ── 정보 박스 (Functional API) ── */
.info-box {
    background: rgba(16, 185, 129, 0.06);
    border-left: 4px solid #10B981;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text-color);
}

/* ── 예측 결과 테이블 ── */
.pred-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.8rem;
    font-size: 0.9rem;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}
.pred-table th {
    background: rgba(79, 70, 229, 0.12);
    color: #4F46E5;
    padding: 0.5rem 0.8rem;
    text-align: left;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.pred-table td {
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid rgba(128,128,128,0.12);
    color: var(--text-color);
}
.pred-table tr:hover td {
    background: rgba(79,70,229,0.04);
}

/* ── 순위 뱃지 ── */
.rank-badge {
    display: inline-block; width: 22px; height: 22px;
    border-radius: 50%; text-align: center;
    line-height: 22px; font-size: 0.75rem; font-weight: 700;
}
.rank-1 { background: #4F46E5; color: #fff; }
.rank-2 { background: #EC4899; color: #fff; }
.rank-3 { background: #10B981; color: #fff; }

/* ── 확률 바 ── */
.prob-bar-wrap {
    background: rgba(128,128,128,0.12);
    border-radius: 999px;
    height: 7px;
    min-width: 80px;
}
.prob-bar {
    height: 7px;
    border-radius: 999px;
    background: linear-gradient(90deg, #4F46E5, #EC4899);
}

/* ── 레이어 라벨 뱃지 ── */
.layer-label {
    display: inline-block;
    background: rgba(79,70,229,0.10);
    color: #4F46E5;
    border: 1px solid rgba(79,70,229,0.35);
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.82rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# ③ 모델 로딩 (@st.cache_resource — 최초 1회만 다운로드)
# =============================================================================
@st.cache_resource
def load_vgg16():
    """VGG16(ImageNet)을 로드합니다. 캐시되어 재실행 시 재사용됩니다."""
    return VGG16(weights='imagenet')

# =============================================================================
# ④ 이미지 전처리
# =============================================================================
def preprocess_image(img: Image.Image) -> np.ndarray:
    """PIL Image → VGG16 전용 전처리 numpy 배열 (1, 224, 224, 3)"""
    img_resized = img.resize((224, 224))
    arr = np.expand_dims(np.array(img_resized), axis=0)
    return preprocess_input(arr)

# ── 로컈 샘플 이미지 경로 정의 ──
# app.py는 DL005/ 안에 있으므로, 상위 DL/ 디렉터리를 참조
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # DL005/
SAMPLE_IMAGES = {
    "🐶 강아지": os.path.join(BASE_DIR, "..", "강아지.jpg"),
    "🐱 고양이": os.path.join(BASE_DIR, "..", "고양이.jpg"),
}

# =============================================================================
# ⑥ Feature Map 시각화
# =============================================================================
def plot_feature_maps(feature_maps: np.ndarray, layer_name: str, num_channels: int = 16):
    """
    Feature Map을 격자(grid)로 시각화합니다.
    feature_maps: shape (1, H, W, C)
    """
    fmaps = feature_maps[0]                          # (H, W, C)
    n_show = min(num_channels, fmaps.shape[-1])
    n_cols = 4
    n_rows = math.ceil(n_show / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.8),
                             facecolor='#0f0c29')
    fig.suptitle(f"{layer_name}  ({n_show}채널)",
                 color='#c4b5fd', fontsize=11, fontweight='bold', y=1.01)

    axes_flat = np.array(axes).flatten() if n_show > 1 else [axes]
    for i, ax in enumerate(axes_flat):
        if i < n_show:
            ax.imshow(fmaps[:, :, i], cmap='viridis', aspect='auto')
            ax.set_title(f"ch {i}", color='#93c5fd', fontsize=7, pad=2)
        ax.axis('off')

    plt.tight_layout(pad=0.4)
    return fig

# =============================================================================
# ⑦ 레이어 목록
# =============================================================================
CONV_LAYERS = [
    "block1_conv1", "block1_conv2",
    "block2_conv1", "block2_conv2",
    "block3_conv1", "block3_conv2", "block3_conv3",
    "block4_conv1", "block4_conv2", "block4_conv3",
    "block5_conv1", "block5_conv2", "block5_conv3",
]

# =============================================================================
# ⑧ 헤더 타이틀
# =============================================================================
st.markdown("""
<div class="main-title">
    <h1>🌌 Sequential  vs  Functional API</h1>
    <p>VGG16 내부를 들여다보다 — 좌측: 최종 예측(Sequential) &nbsp;|&nbsp; 우측: 중간 Feature Map(Functional API)</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# ⑨ 컨트롤 바 — 이미지 업로드 / 샘플 미리보기 / 채널 수 / 모델 정보
#    사이드바 대신 본문 상단에 가로로 배치
# =============================================================================
st.markdown("""
<div class="glass-card" style="margin-bottom:0.6rem;">
    <span style="font-size:0.85rem;opacity:0.7;">
    📂 이미지를 업로드하거나, 아래 샘플 이미지 중 하나를 선택해 바로 테스트하세요.
    </span>
</div>
""", unsafe_allow_html=True)

# 헤더 컨트롤 바: 업로더 | 샘플 선택 | 샘플 썸네일 | 채널 수 | 모델정보
ctrl_upload, ctrl_sample, ctrl_thumb, ctrl_slider, ctrl_info = st.columns(
    [2, 1.4, 1.8, 1.4, 1.2], gap="medium"
)

with ctrl_upload:
    # 이미지 파일 업로드 위젯 (업로드시 샘플 선택은 무시됨)
    uploaded_file = st.file_uploader(
        "🖼️ 이미지 업로드 (jpg/jpeg/png)",
        type=["jpg", "jpeg", "png"],
        help="업로드하면 샘플 선택이 무시되고 업로드 이미지가 사용됩니다.",
        label_visibility="visible",
    )

with ctrl_sample:
    # 샘플 이미지 라디오 선택기
    sample_choice = st.radio(
        "🐾 샘플 선택",
        options=list(SAMPLE_IMAGES.keys()),
        index=1,   # 기본: 고양이
        help="업로드하지 않으면 선택한 샘플이 사용됩니다.",
    )

with ctrl_thumb:
    # 샘플 이미지 썸네일 2개 나란히 표시
    th1, th2 = st.columns(2)
    for col, (label, path) in zip([th1, th2], SAMPLE_IMAGES.items()):
        try:
            thumb_img = Image.open(path).convert("RGB")
            # 선택된 샘플에 하이라이트 테두리 표시
            border = "3px solid #4F46E5" if label == sample_choice and uploaded_file is None else "none"
            col.markdown(
                f"<div style='border:{border};border-radius:8px;overflow:hidden;'></div>",
                unsafe_allow_html=True
            )
            col.image(thumb_img, caption=label, use_container_width=True)
        except FileNotFoundError:
            col.caption(f"{label} 파일 없음")

with ctrl_slider:
    # Feature Map 채널 수 선택
    num_channels = st.select_slider(
        "🎛️ 표시할 채널 수",
        options=[4, 8, 12, 16],
        value=16,
        help="채널 수를 줄이면 더 빠르게 렌더링됩니다.",
    )

with ctrl_info:
    # 모델 메타 정보
    st.markdown("""
    <div style='background:rgba(79,70,229,0.06);border:1px solid rgba(79,70,229,0.2);
                border-radius:10px;padding:0.8rem;font-size:0.82rem;line-height:2;
                margin-top:0.3rem;'>
        ⚙️ <b style='color:#4F46E5;'>VGG16</b> (ImageNet)<br>
        📐 입력: <b>224 × 224</b> px<br>
        🏷️ 클래스: <b>1,000</b>개
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# ⑩ 이미지 로드 우선순위:
#   1순위. 업로드 파일
#   2순위. 라디오로 선택된 로컈 샘플 이미지
# =============================================================================
if uploaded_file is not None:
    # 업로드 우선
    pil_image = Image.open(uploaded_file).convert("RGB")
    src_label  = "📤 업로드 이미지"
else:
    # 라디오에서 선택된 로컈 샘플 사용
    selected_path = SAMPLE_IMAGES[sample_choice]
    try:
        pil_image = Image.open(selected_path).convert("RGB")
        src_label  = f"{sample_choice} (샘플)"
    except FileNotFoundError:
        st.error(
            f"샘플 파일을 찾을 수 없습니다: `{selected_path}`\n"
            f"DL/ 폴더에 **강아지.jpg** 와 **고양이.jpg** 가 있는지 확인하세요."
        )
        st.stop()

# 모델 로드 (캐시 — 최초 1회만 다운로드)
with st.spinner("🔄 VGG16 모델 로딩 중 (최초 1회)..."):
    model = load_vgg16()

preprocessed = preprocess_image(pil_image)

# 입력 이미지 미리보기 (컨트롤 바 아래 좌측)
img_col, _ = st.columns([1, 4])
with img_col:
    st.image(pil_image, caption=src_label, width=200)

st.markdown("---")

# =============================================================================
# ⑫ 좌우 2-Column 레이아웃
#    LEFT  : Sequential — 최종 예측 결과
#    RIGHT : Functional API — 레이어 선택 + Feature Map
# =============================================================================
col_left, col_right = st.columns([1, 1], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# 좌측 패널 — Sequential 모델
# ─────────────────────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div class="panel-header-seq">
        <h2>📌 Sequential 모델</h2>
        <p>model.predict() → 최종 출력(1000-class softmax)만 반환</p>
    </div>
    """, unsafe_allow_html=True)

    # VGG16 모델 구조 간략 표시
    with st.expander("📋 VGG16 레이어 구조 보기", expanded=False):
        rows = []
        for i, layer in enumerate(model.layers):
            # 최신 Keras에서 InputLayer는 output_shape 속성이 없으므로 안전하게 처리
            try:
                shape_str = str(layer.output_shape)
            except AttributeError:
                try:
                    shape_str = str(layer.output.shape)
                except Exception:
                    shape_str = "(입력 레이어)"
            rows.append(f"| `{layer.name}` | `{shape_str}` |")
        table_md  = "| 레이어 이름 | 출력 Shape |\n|---|---|\n"
        table_md += "\n".join(rows[:14])
        table_md += f"\n| ... | ... |\n| *(총 {len(model.layers)}개)* | |"
        st.markdown(table_md)

    # Predict 버튼
    predict_btn = st.button(
        "🚀 Predict — 최종 결과 확인", key="btn_seq", use_container_width=True
    )

    if predict_btn:
        with st.spinner("예측 중..."):
            # ★ Sequential 모델: input → output (최종 softmax 1000)
            preds   = model.predict(preprocessed)          # (1, 1000)
            decoded = decode_predictions(preds, top=3)[0]  # 상위 3개 디코딩

        rank_badges = [
            '<span class="rank-badge rank-1">1</span>',
            '<span class="rank-badge rank-2">2</span>',
            '<span class="rank-badge rank-3">3</span>',
        ]
        rows_html = ""
        for i, (_, label, prob) in enumerate(decoded):
            bar_w = int(prob * 100)
            rows_html += f"""
            <tr>
                <td>{rank_badges[i]}</td>
                <td><b>{label.replace('_',' ')}</b></td>
                <td style='text-align:right;'><b>{prob*100:.1f}%</b></td>
                <td style='width:90px;'>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar" style="width:{bar_w}%;"></div>
                    </div>
                </td>
            </tr>"""

        st.markdown(f"""
        <div class="glass-card">
            <b style="color:#4F46E5;font-size:1rem;">🏆 Top-3 예측 결과</b>
            <table class="pred-table">
                <thead><tr>
                    <th>순위</th><th>클래스</th><th>확률</th><th>비율</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # 한계 안내 메시지
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Sequential 모델의 한계</b><br>
    Sequential 모델은 이처럼 <b>최종 결과만 반환</b>하므로,
    내부에서 무슨 일이 일어나는지 알 수 없습니다.<br>
    모델이 엣지·텍스쳐·형태 등 어떤 특징을 학습했는지 확인하려면
    <b>Functional API</b>가 필요합니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📖 Sequential 예측 코드**")
    st.code("""
# Sequential(VGG16) 모델로 최종 예측
preds   = model.predict(img_array)
decoded = decode_predictions(preds, top=3)[0]
""", language="python")


with col_right:
    st.markdown("""
    <div class="panel-header-func">
        <h2>🔗 Functional API</h2>
        <p>tf.keras.Model(inputs, outputs) → 중간 레이어 Feature Map 추출</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    ✅ Functional API를 사용하면 <b>기존 모델의 가중치를 공유</b>하는 임시 모델을 만들어
    중간 출력을 가로채을 수 있습니다.<br>
    <code>tf.keras.Model(inputs=model.input, outputs=model.get_layer(name).output)</code>
    </div>
    """, unsafe_allow_html=True)

    layer_idx = st.slider(
        "🎚️ 시각화할 Conv 레이어",
        min_value=0,
        max_value=len(CONV_LAYERS) - 1,
        value=0,
        format="%d",
    )
    selected_layer = CONV_LAYERS[layer_idx]

    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:0.8rem;margin:0.4rem 0 0.8rem;'>
        <span class="layer-label">📍 {selected_layer}</span>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"⚙️ `{selected_layer}` Feature Map 추출 중..."):
        feature_model = tf.keras.Model(
            inputs  = model.input,
            outputs = model.get_layer(selected_layer).output
        )
        feature_maps = feature_model.predict(preprocessed)

    h, w, c = feature_maps.shape[1], feature_maps.shape[2], feature_maps.shape[3]
    m1, m2, m3 = st.columns(3)
    m1.metric("📏 Feature Map 높이", f"{h}px")
    m2.metric("📏 Feature Map 너비", f"{w}px")
    m3.metric("🌐 전체 채널 수", f"{c}개")

    st.markdown(f"**🗺️ Feature Map 시각화 — `{selected_layer}` ({num_channels}채널)**")
    fig = plot_feature_maps(feature_maps, selected_layer, num_channels=num_channels)
    st.pyplot(fig, use_container_width=True)

    # 핵심 코드 스니펫
    st.markdown("**📖 Functional API 핵심 코드**")
    st.code(f"""
# ★ 기존 VGG16 가중치를 공유하는 임시 모델 생성
feature_model = tf.keras.Model(
    inputs  = model.input,                              # 원본 입력 텐서
    outputs = model.get_layer('{selected_layer}').output  # 중간 레이어 출력
)

# 이미지 통과 → Feature Map 추출
feature_maps = feature_model.predict(img_array)
# shape: (1, {h}, {w}, {c})

# 특정 채널 시각화
plt.imshow(feature_maps[0, :, :, 0], cmap='viridis')
plt.axis('off')
plt.show()
""", language="python")

# =============================================================================
# ⑬ 하단 비교 요약 표
# =============================================================================
st.markdown("---")
st.markdown("""<div style='font-size:1.4rem;font-weight:800;
    background:linear-gradient(135deg,#4F46E5,#EC4899);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    margin-bottom:0.6rem;'>📊 Sequential vs Functional API 핵심 비교</div>""",
    unsafe_allow_html=True)
st.markdown("""
| 항목 | 📌 Sequential 모델 | 🔗 Functional API |
|---|---|---|
| **구조** | 레이어가 선형으로 연결 | 입력/출력 텐서를 자유롭게 연결 |
| **출력** | 최종 레이어 결과만 반환 | **원하는 중간 레이어** 출력 추출 가능 |
| **내부 시각화** | ❌ 불가 | ✅ Feature Map 추출 가능 |
| **가중치** | 원본 모델 학습 가중치 사용 | 원본 가중치 **공유** (재학습 불필요) |
| **응용** | 일반 분류/회귀 | Transfer Learning, GAN, U-Net 등 |
""")

# =============================================================================
# ⑭ 푸터
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#4a5568;font-size:0.78rem;padding:0.8rem 0;'>
    🔬 Sequential vs Functional API Visualizer &nbsp;|&nbsp;
    Built with Streamlit + TensorFlow/Keras &nbsp;|&nbsp; Model: VGG16 (ImageNet)
</div>
""", unsafe_allow_html=True)
