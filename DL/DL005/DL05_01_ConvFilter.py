import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import convolve2d
from PIL import Image
import platform

# Matplotlib 한글 폰트 설정 (Windows: Malgun Gothic, macOS: AppleGothic, Linux: NanumGothic)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 1. Page Configuration (Handles both standalone and multi-page wrapper execution)
try:
    st.set_page_config(
        page_title="CNN 필터 시각화 및 특성 맵 시뮬레이터",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except st.errors.StreamlitAPIException:
    pass

# Custom CSS Styles for High-End Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header and Typography */
    .title-text {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4F46E5 0%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.8;
        margin-bottom: 1.5rem;
    }
    
    .section-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 15px;
        border-left: 5px solid #4F46E5;
        padding-left: 12px;
        color: var(--text-color);
    }

    /* Metric Cards Style */
    .stat-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .stat-card {
        flex: 1;
        background: var(--background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 14px;
        padding: 18px 24px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        border-color: #4F46E5;
    }
    
    .stat-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-color);
        line-height: 1.1;
    }
    
    .stat-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .delta-plus {
        color: #10B981; /* Emerald */
    }
    
    .delta-minus {
        color: #EF4444; /* Rose */
    }
    
    .delta-neutral {
        color: var(--text-color);
        opacity: 0.5;
    }

    /* HTML Grid Matrix Style */
    .matrix-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .matrix-cell {
        width: 33.33%;
        height: 55px;
        text-align: center;
        font-weight: 700;
        font-size: 0.95rem;
        border: 2px solid var(--background-color);
        transition: background-color 0.2s;
    }
</style>
""", unsafe_allow_html=True)

# 2. Mock Data Generator with Cache
@st.cache_data
def generate_mock_weights():
    # Seed for deterministic and reproducible mock data
    np.random.seed(42)
    
    # Untrained Weights: (3, 3, 1, 32)
    # Target: mean=0.005, std=0.085 (Washed out, low-contrast noise)
    untrained_raw = np.random.normal(loc=0.005, scale=0.085, size=(3, 3, 1, 32))
    
    # Trained Weights: (3, 3, 1, 32)
    # Target: mean=-0.018, std=0.227 (High contrast, features emerge)
    trained_raw = np.random.normal(loc=-0.018, scale=0.227, size=(3, 3, 1, 32))
    
    # Specific edge pattern templates to inject
    # Filter 0: Vertical Edge Detector
    v_edge = np.array([[-1.0, 0.0, 1.0],
                       [-1.0, 0.0, 1.0],
                       [-1.0, 0.0, 1.0]])
    
    # Filter 1: Horizontal Edge Detector
    h_edge = np.array([[-1.0, -1.0, -1.0],
                       [ 0.0,  0.0,  0.0],
                       [ 1.0,  1.0,  1.0]])
    
    # Filter 2: Diagonal Edge Detector
    d_edge = np.array([[-1.0, -1.0,  0.0],
                       [-1.0,  0.0,  1.0],
                       [ 0.0,  1.0,  1.0]])
    
    # Overlay these onto filters 0, 1, 2 (using a scale of 0.8 to make them prominent)
    trained_raw[:, :, 0, 0] = v_edge * 0.8
    trained_raw[:, :, 0, 1] = h_edge * 0.8
    trained_raw[:, :, 0, 2] = d_edge * 0.8
    
    # Overlay other structured edge templates for visual realism
    templates = [
        np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]),  # Vertical (opposite)
        np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]),  # Horizontal (opposite)
        np.array([[0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, -1.0, 0.0]]),  # Diagonal (opposite)
        np.array([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]]),  # Spot/High-pass
        np.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]])  # Spot (inverted)
    ]
    
    for i in range(3, 32):
        tpl = templates[i % len(templates)]
        noise = np.random.normal(0, 0.15, (3, 3))
        # Mix template with noise to represent real weight adjustments
        trained_raw[:, :, 0, i] = tpl * 0.5 + noise
        
    # Standardize untrained exactly to target stats
    untrained = (untrained_raw - np.mean(untrained_raw)) / (np.std(untrained_raw) + 1e-9)
    untrained = untrained * 0.085 + 0.005
    
    # Standardize trained exactly to target stats
    trained = (trained_raw - np.mean(trained_raw)) / (np.std(trained_raw) + 1e-9)
    trained = trained * 0.227 - 0.018
    
    return untrained, trained

# Create default virtual image
def get_default_image():
    # 200x200 canvas
    img = np.zeros((200, 200), dtype=np.float32) + 0.1
    
    # 1. Vertical stripe (left side)
    img[20:180, 35:50] = 0.9
    
    # 2. Horizontal stripe (top-right side)
    img[35:50, 80:165] = 0.9
    
    # 3. Diagonal stripe (center to bottom-right)
    for i in range(200):
        if 80 <= i < 165:
            # Draw diagonal stripe
            img[i, i-10:i+5] = 0.9
            
    # 4. Circle (bottom-left)
    yy, xx = np.ogrid[:200, :200]
    circle_mask = (yy - 135)**2 + (xx - 60)**2 <= 22**2
    img[circle_mask] = 0.9
    
    # 5. Square boundary
    square_mask = (yy >= 105) & (yy <= 155) & (xx >= 120) & (xx <= 170)
    img[square_mask] = 0.9
    square_inner = (yy >= 115) & (yy <= 145) & (xx >= 130) & (xx <= 160)
    img[square_inner] = 0.1
    
    return np.clip(img, 0.0, 1.0)

# Load filter data
untrained_data, trained_data = generate_mock_weights()

# --- MAIN HEADER ---
st.markdown('<div class="title-text">🌌 CNN 필터 가중치 & 특성 맵 시각화 시뮬레이터</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">합성곱 신경망(CNN)의 1번째 레이어 필터들이 학습 전(무작위 초기화)과 학습 후에 어떻게 변화하는지, 그리고 입력 이미지에 실시간으로 어떻게 반응하여 특징을 추출하는지 탐색해 보세요.</div>', unsafe_allow_html=True)

# --- SECTION 1: USER IMAGE INPUT & PROCESSING ---
st.markdown('<div class="section-title">🖼️ 1. 사용자 이미지 입력 및 전처리</div>', unsafe_allow_html=True)

col_up1, col_up2 = st.columns([7, 5], gap="large")

with col_up1:
    st.markdown("""
    CNN 필터가 반응을 보일 대상 이미지를 입력받는 섹션입니다. 
    사용자가 직접 JPG나 PNG 이미지를 업로드할 수 있으며, 업로드하지 않을 시에는 다양한 기하학적 형태(세로선, 가로선, 대각선, 원, 사각형)가 조화롭게 배치된 고대비 가상 테스트 이미지가 기본으로 제공됩니다.
    
    **전처리 과정 (Pre-processing)**
    1. 합성곱 연산의 편의성과 속도를 위해 모든 이미지는 **흑백(Grayscale)**으로 변환됩니다.
    2. 실시간 연산 지연을 방지하기 위해 가로세로 **200x200 해상도**로 즉시 리사이징됩니다.
    """)
    
    uploaded_file = st.file_uploader(
        "📥 이미지 업로드 (JPG, PNG)", 
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

with col_up2:
    # Process image
    if uploaded_file is not None:
        try:
            pil_img = Image.open(uploaded_file)
            # Convert to grayscale
            gray_img = pil_img.convert("L")
            # Resize
            resized_img = gray_img.resize((200, 200))
            img_arr = np.array(resized_img, dtype=np.float32) / 255.0
            st.success("📸 사용자 이미지가 업로드되어 200x200 Grayscale로 변환되었습니다.")
        except Exception as e:
            st.error(f"이미지 처리 중 오류가 발생했습니다. 기본 가상 이미지를 사용합니다. ({e})")
            img_arr = get_default_image()
    else:
        img_arr = get_default_image()
        st.info("💡 기본 제공 고대비 가상 이미지(Geometric Pattern)가 활성화되어 있습니다.")

    # Image Preview inside columns
    prev_col1, prev_col2 = st.columns([1, 1.5])
    with prev_col1:
        st.image(img_arr, caption="전처리 완료된 입력 이미지", use_container_width=True)
    with prev_col2:
        # Mini stats on input image
        st.markdown(f"""
        **이미지 명세 (Image Specs)**
        - **해상도**: 200 × 200 픽셀 (2D 단일 채널)
        - **데이터 타입**: `float32`
        - **값 범위**: `[{img_arr.min():.2f}, {img_arr.max():.2f}]` (0은 검은색, 1은 흰색)
        """)


# --- SECTION 2: LEFT/RIGHT FILTER GRID COMPARISON ---
st.markdown('<div class="section-title">📊 2. 학습 전 / 후 필터 가중치 그리드 대조</div>', unsafe_allow_html=True)

st.markdown("""
첫 번째 합성곱 레이어(Conv2D)에 속한 **32개의 필터 가중치(3x3 크기)**를 학습 전(무작위 초기값)과 학습 후로 나누어 전수 비교해봅니다.
가중치의 미세한 대비 변화를 직관적으로 비교할 수 있도록 색상 범위는 **`vmin=-0.5, vmax=0.5`**로 완벽하게 고정되어 있습니다.

- **학습 전 (Random Initialization)**: 정규분포(평균 0.005, 표준편차 0.085)를 따르는 밋밋하고 평평한 무작위 노이즈 상태입니다.
- **학습 후 (Trained)**: 정규분포(평균 -0.018, 표준편차 0.227)로 표준편차가 **약 3배** 커지며, 필터 고유의 방향성(가로, 세로, 대각선 에지 등)에 맞추어 양극단으로 분화된 뚜렷한 대비를 보여줍니다.
""")

# Sidebar controls for customization
st.sidebar.markdown("### ⚙️ 시뮬레이터 글로벌 설정")
cmap_choice = st.sidebar.selectbox(
    "🎨 가중치 시각화 색상 맵 (Colormap)",
    options=["coolwarm", "gray", "viridis"],
    index=0
)

# Highlight indicator
if "selected_filter" not in st.session_state:
    st.session_state.selected_filter = 0

selected_filter = st.sidebar.slider(
    "🔍 상세 분석할 필터 인덱스 (0 ~ 31)",
    min_value=0,
    max_value=31,
    value=st.session_state.selected_filter,
    key="selected_filter_slider"
)
st.session_state.selected_filter = selected_filter

col_grid1, col_grid2 = st.columns(2, gap="large")

# Common function to render 4x8 filter grid
def plot_grid(weights, title, selected_idx, cmap):
    fig, axes = plt.subplots(4, 8, figsize=(10, 5.5))
    fig.patch.set_alpha(0.0) # Transparent background
    
    for i in range(32):
        ax = axes[i // 8, i % 8]
        ax.patch.set_alpha(0.0)
        filter_matrix = weights[:, :, 0, i]
        
        # Plot weights image
        im = ax.imshow(filter_matrix, cmap=cmap, vmin=-0.5, vmax=0.5, interpolation='nearest')
        ax.set_title(f"Idx {i}", fontsize=9, color="#888888", pad=2)
        ax.axis('off')
        
        # Highlight Selected Filter
        if i == selected_idx:
            rect = plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor='#FF4B4B', linewidth=3, clip_on=False)
            ax.add_patch(rect)
            
    plt.tight_layout(pad=0.5)
    return fig

with col_grid1:
    st.markdown("#### 🚫 [좌측] 학습 전 필터 가중치 (Before Training)")
    fig_untrained = plot_grid(untrained_data, "Before Training", selected_filter, cmap_choice)
    st.pyplot(fig_untrained, use_container_width=True)
    plt.close(fig_untrained)
    
    # Statistical Info
    mean_un = np.mean(untrained_data)
    std_un = np.std(untrained_data)
    st.markdown(f"""
    <div style="background: rgba(128, 128, 128, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(128,128,128,0.15);">
        <strong>학습 전 요약 통계치 (Target: mean=0.005, std=0.085)</strong><br>
        • 실제 평균(Mean): <code>{mean_un:.5f}</code><br>
        • 실제 표준편차(Std Dev): <code>{std_un:.5f}</code><br>
        • 필터 특징: 3x3 격자 내 값이 서로 비슷하여 시각화 시 대부분 회색빛을 띱니다.
    </div>
    """, unsafe_allow_html=True)

with col_grid2:
    st.markdown("#### 🎯 [우측] 학습 후 필터 가중치 (After Training)")
    fig_trained = plot_grid(trained_data, "After Training", selected_filter, cmap_choice)
    st.pyplot(fig_trained, use_container_width=True)
    plt.close(fig_trained)
    
    # Statistical Info
    mean_tr = np.mean(trained_data)
    std_tr = np.std(trained_data)
    st.markdown(f"""
    <div style="background: rgba(79, 70, 229, 0.05); padding: 12px; border-radius: 8px; border: 1px solid rgba(79, 70, 229, 0.15);">
        <strong>학습 후 요약 통계치 (Target: mean=-0.018, std=0.227)</strong><br>
        • 실제 평균(Mean): <code>{mean_tr:.5f}</code><br>
        • 실제 표준편차(Std Dev): <code>{std_tr:.5f}</code><br>
        • 필터 특징: 붉은색(+)과 푸른색(-)의 격렬한 보색 대비가 나타나며, 엣지 감지 기능이 부여되었습니다.
    </div>
    """, unsafe_allow_html=True)


# --- SECTION 3: INTERACTIVE FILTER RESPONSE SIMULATOR ---
st.markdown('<div class="section-title">🔍 3. 인터랙티브 필터 반응 시뮬레이터 (Feature Map Linkage)</div>', unsafe_allow_html=True)

st.markdown("""
분석하고자 하는 특정 필터 번호를 선택하세요. 선택된 필터의 확대된 가중치 수치(Annotated Heatmap)와 
해당 필터를 입력 이미지에 적용한 실시간 합성곱(Convolution) 연산 결과인 **특성 맵(Feature Map)**을 나란히 관찰할 수 있습니다.
""")

# Selector with helper label
filter_options = list(range(32))
def get_filter_label(idx):
    if idx == 0:
        return "Filter 0 (★강조★ 세로 에지 감지 필터)"
    elif idx == 1:
        return "Filter 1 (★강조★ 가로 에지 감지 필터)"
    elif idx == 2:
        return "Filter 2 (★강조★ 대각선 에지 감지 필터)"
    elif idx % 5 == 3:
        return f"Filter {idx} (반대 방향 세로/가로 에지 필터)"
    elif idx % 5 == 0:
        return f"Filter {idx} (스팟/점 에지 감지 필터)"
    else:
        return f"Filter {idx} (일반 기하학 필터)"

# Sync selected filter from selector
selected_filter = st.selectbox(
    "🎨 관심 필터 번호 선택 (Filter Index Selector)",
    options=filter_options,
    index=st.session_state.selected_filter,
    format_func=get_filter_label,
    key="selectbox_filter_bottom"
)
st.session_state.selected_filter = selected_filter

# Activation function & color scale options
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    activation_mode = st.radio(
        "⚡ 활성화 함수 (Activation Function) 적용 여부",
        options=["Raw Linear (활성화 없음)", "ReLU (음수 활성화 0 처리)", "Absolute Value (절대값 에지 강조)"],
        index=1, # ReLU as default
        horizontal=True
    )
with col_opt2:
    scale_mode = st.radio(
        "⚖️ 특성 맵 시각화 스케일 모드 (Color Scale Mode)",
        options=["공동 스케일 (학습 전/후의 절대적 반응 강도 비교)", "개별 스케일 (특징 패턴 형상에 집중)"],
        index=0,
        horizontal=True
    )

# Slice selected filter weights: shape (3,3)
w_un = untrained_data[:, :, 0, selected_filter]
w_tr = trained_data[:, :, 0, selected_filter]

# convolve2d operations
fmap_un_raw = convolve2d(img_arr, w_un, mode='same', boundary='symm')
fmap_tr_raw = convolve2d(img_arr, w_tr, mode='same', boundary='symm')

# Apply Activation
def apply_activation(fmap, mode):
    if mode == "ReLU (음수 활성화 0 처리)":
        return np.maximum(0, fmap)
    elif mode == "Absolute Value (절대값 에지 강조)":
        return np.abs(fmap)
    else:
        return fmap

fmap_un = apply_activation(fmap_un_raw, activation_mode)
fmap_tr = apply_activation(fmap_tr_raw, activation_mode)

# Calculate rendering limits
if scale_mode == "공동 스케일 (학습 전/후의 절대적 반응 강도 비교)":
    f_vmin = min(fmap_un.min(), fmap_tr.min())
    f_vmax = max(fmap_un.max(), fmap_tr.max())
    # Ensure some spread to avoid divide by zero if maps are identical
    if abs(f_vmax - f_vmin) < 1e-5:
        f_vmin, f_vmax = 0.0, 1.0
else:
    f_vmin, f_vmax = None, None

bottom_col1, bottom_col2 = st.columns([5, 7], gap="large")

with bottom_col1:
    st.markdown("##### 🧮 3x3 가중치 수치 행렬 (Annotated Heatmap)")
    
    fig_w, (ax_w_un, ax_w_tr) = plt.subplots(1, 2, figsize=(7, 3.5))
    fig_w.patch.set_alpha(0.0)
    
    # Left: Untrained weight heatmap
    ax_w_un.patch.set_alpha(0.0)
    ax_w_un.imshow(w_un, cmap=cmap_choice, vmin=-0.5, vmax=0.5, interpolation='nearest')
    ax_w_un.set_title("학습 전 (Untrained)", fontsize=10, pad=10, color="#888888")
    for r in range(3):
        for c in range(3):
            val = w_un[r, c]
            # Use dynamic text color for legibility
            n_val = (val - (-0.5)) / (0.5 - (-0.5) + 1e-9)
            n_val = max(0.0, min(1.0, n_val))
            rgb = cm.get_cmap(cmap_choice)(n_val)
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = "black" if lum > 0.5 else "white"
            ax_w_un.text(c, r, f"{val:+.3f}", ha='center', va='center', color=text_color, fontweight='bold', fontsize=10)
    ax_w_un.set_xticks([0, 1, 2])
    ax_w_un.set_yticks([0, 1, 2])
    ax_w_un.set_xticklabels(['C0', 'C1', 'C2'], color='#888888', fontsize=8)
    ax_w_un.set_yticklabels(['R0', 'R1', 'R2'], color='#888888', fontsize=8)
    ax_w_un.tick_params(colors='#888888')
    ax_w_un.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax_w_un.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax_w_un.grid(which='minor', color='white', linestyle='-', linewidth=2)

    # Right: Trained weight heatmap
    ax_w_tr.patch.set_alpha(0.0)
    ax_w_tr.imshow(w_tr, cmap=cmap_choice, vmin=-0.5, vmax=0.5, interpolation='nearest')
    ax_w_tr.set_title("학습 후 (Trained)", fontsize=10, pad=10, color="#888888")
    for r in range(3):
        for c in range(3):
            val = w_tr[r, c]
            n_val = (val - (-0.5)) / (0.5 - (-0.5) + 1e-9)
            n_val = max(0.0, min(1.0, n_val))
            rgb = cm.get_cmap(cmap_choice)(n_val)
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = "black" if lum > 0.5 else "white"
            ax_w_tr.text(c, r, f"{val:+.3f}", ha='center', va='center', color=text_color, fontweight='bold', fontsize=10)
    ax_w_tr.set_xticks([0, 1, 2])
    ax_w_tr.set_yticks([0, 1, 2])
    ax_w_tr.set_xticklabels(['C0', 'C1', 'C2'], color='#888888', fontsize=8)
    ax_w_tr.set_yticklabels(['R0', 'R1', 'R2'], color='#888888', fontsize=8)
    ax_w_tr.tick_params(colors='#888888')
    ax_w_tr.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax_w_tr.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax_w_tr.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    st.pyplot(fig_w, use_container_width=True)
    plt.close(fig_w)

    # Print markdown table details
    st.markdown("###### 🔍 학습 후 3x3 가중치 행렬 세부 수치")
    html_table = '<table class="matrix-table">'
    for r in range(3):
        html_table += '<tr>'
        for c in range(3):
            val = w_tr[r, c]
            n_val = (val - (-0.5)) / (0.5 - (-0.5) + 1e-9)
            n_val = max(0.0, min(1.0, n_val))
            rgba = cm.get_cmap(cmap_choice)(n_val)
            r_val, g_val, b_val = [int(x * 255) for x in rgba[:3]]
            lum = (0.299 * r_val + 0.587 * g_val + 0.114 * b_val) / 255.0
            text_color = "#000000" if lum > 0.5 else "#FFFFFF"
            html_table += f'<td class="matrix-cell" style="background-color: rgb({r_val},{g_val},{b_val}); color: {text_color};">{val:+.4f}</td>'
        html_table += '</tr>'
    html_table += '</table>'
    st.markdown(html_table, unsafe_allow_html=True)

with bottom_col2:
    st.markdown("##### ⚡ 실시간 합성곱 연산 결과 (Feature Map Response)")
    
    fig_f, (ax_f_un, ax_f_tr) = plt.subplots(1, 2, figsize=(9, 4.5))
    fig_f.patch.set_alpha(0.0)
    
    # Untrained Feature Map
    ax_f_un.patch.set_alpha(0.0)
    im_un = ax_f_un.imshow(fmap_un, cmap='gray', vmin=f_vmin, vmax=f_vmax)
    ax_f_un.set_title("학습 전 특성 맵 (Before Training)", fontsize=10, pad=10, color="#888888")
    ax_f_un.axis('off')
    
    # Trained Feature Map
    ax_f_tr.patch.set_alpha(0.0)
    im_tr = ax_f_tr.imshow(fmap_tr, cmap='gray', vmin=f_vmin, vmax=f_vmax)
    ax_f_tr.set_title("학습 후 특성 맵 (After Training)", fontsize=10, pad=10, color="#888888")
    ax_f_tr.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig_f, use_container_width=True)
    plt.close(fig_f)
    
    # Description of results based on filter selection
    if selected_filter == 0:
        effect_msg = "💡 **수직 에지 필터(Filter 0) 관찰**: 학습 후 특성 맵에서 입력 이미지의 **세로선(수직 경계선)**들이 밝은 백색으로 뚜렷하게 부각되는 것을 볼 수 있습니다. 반면 학습 전 특성 맵은 거의 무반응에 가까운 흐릿한 노이즈만 보입니다."
    elif selected_filter == 1:
        effect_msg = "💡 **수평 에지 필터(Filter 1) 관찰**: 학습 후 특성 맵에서 입력 이미지의 **가로선(수평 경계선)**들이 아주 강하게 검출되어 하얗게 빛나는 것을 볼 수 있습니다. 학습 전 특성 맵은 이러한 구조 정보를 전혀 잡지 못합니다."
    elif selected_filter == 2:
        effect_msg = "💡 **대각선 에지 필터(Filter 2) 관찰**: 학습 후 특성 맵에서 입력 이미지의 **대각선 경계선**이 선명하게 하이라이트됩니다. 학습 전 필터는 단지 형태가 뭉개진 저대비 노이즈 패턴만 생성합니다."
    else:
        effect_msg = "💡 **기타 기하학적 필터 관찰**: 학습을 거치면서 필터 가중치가 점이나 복합적인 엣지를 탐색하도록 훈련되었으며, 이에 대응하는 이미지 부위에서 유의미한 합성곱 신호(백색 활성화)를 검출해 내는 것을 볼 수 있습니다."
        
    st.info(effect_msg)

# --- EDUCATIONAL EXPANDER PANEL ---
st.markdown("<br><hr>", unsafe_allow_html=True)
with st.expander("💡 CNN 특성 맵 실시간 반응 원리 설명", expanded=True):
    st.markdown("""
    ### 1. 합성곱 연산(Convolution)의 동작 원리
    합성곱 연산은 이미지 위를 3x3 필터(커널)가 슬라이딩하면서 겹치는 영역의 픽셀 값들을 곱하고 모두 더해 새로운 픽셀을 만드는 연산입니다:
    $$\\text{Feature Map}(y, x) = \\sum_{i=-1}^{1} \\sum_{j=-1}^{1} \\text{Image}(y+i, x+j) \\times \\text{Filter}(i+1, j+1)$$
    
    ### 2. 학습 전 vs 학습 후 반응 차이가 극명한 이유
    - **학습 전 필터 (가중치 작음, 무작위)**: 
        - 필터의 각 픽셀이 0에 매우 가깝고 무작위로 섞여 있어, 연산 결과가 평균적으로 서로 상쇄됩니다.
        - 따라서 이미지 상의 어떠한 유의미한 형태나 경계선 정보도 검출하지 못하고 흐릿한 그레이스케일 노이즈만 남습니다.
    - **학습 후 필터 (가중치 큼, 특정 엣지 패턴 획득)**:
        - 훈련을 거치며 세로선(Filter 0), 가로선(Filter 1), 대각선(Filter 2) 등 입력 이미지에서 가장 중요한 정보인 '경계(Edge)'를 추출할 수 있는 **값의 배치**를 이룩하였습니다.
        - 이미지에서 해당 경계를 만나게 되면 합성곱 값이 증폭되어 **매우 큰 양수(또는 음수) 반응**을 얻으며, 이것이 특성 맵 상에서 아주 밝은 흰색 선으로 시각화되는 것입니다.

    ### 3. 활성화 함수(ReLU)의 기여
    - 실제 CNN 모델에서는 합성곱 연산 직후 활성화 함수(주로 **ReLU**)를 거칩니다.
    - ReLU는 음수 값을 0(검은색)으로 클리핑하여 돌려주며, 오직 양수 반응이 나타나는 핵심 엣지 정보만을 다음 레이어로 전달하는 비선형 필터링 역할을 담당합니다.
    - 활성화 함수를 **Raw Linear**에서 **ReLU**로 바꾸어가며 음수 반응(보통 경계의 반대 방향이나 어두워지는 경계)이 어떻게 차단되는지 관찰해 보세요!
    """)
