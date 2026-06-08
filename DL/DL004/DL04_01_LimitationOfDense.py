import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# ==============================================================================
# INITIALIZE SESSION STATE (For Language and State persistence)
# ==============================================================================
if 'lang' not in st.session_state:
    st.session_state.lang = 'KOR'

if 'flattened' not in st.session_state:
    st.session_state.flattened = False

# ==============================================================================
# PAGE CONFIGURATION & RICH THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Limitations of Dense Layers: The Flattening Problem",
    page_icon="🧩",
    layout="wide"
)

# Custom CSS to apply premium typography (Outfit font), header layout, and dynamic styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #2A2A4E;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #AB63FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #8888AA;
    }
    .metric-container {
        background-color: #1E1E2F;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #3A3A5A;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-val-2d {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FF4B4B;
    }
    .metric-val-1d {
        font-size: 2.8rem;
        font-weight: 800;
        color: #00F5FF;
    }
    .callout-card {
        border-left: 5px solid #AB63FA;
        background-color: #1A1A2E;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #2A2A4E;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# TRANSLATIONS DICTIONARY (Bilingual Support)
# ==============================================================================
t = {
    'ENG': {
        'title': "Limitations of Dense (Flatten) Layers",
        'subtitle': "An Interactive Guide to the Loss of Spatial Information in Neural Networks",
        'sec1_header': "1. Preservation of Spatial Relationships",
        'sec1_p1': """In visual tasks (like image classification or object detection), **images are not just arbitrary sets of pixel colors; they contain inherent structure.**
Pixels that are physically close to each other form edges, textures, shapes, and complex features. For instance, the pixels representing a person's eye, nose, or mouth are always clustered together in a specific spatial neighborhood.""",
        'analogy_title': "🧩 The Jigsaw Puzzle Analogy",
        'analogy_body': """Imagine taking a completed **jigsaw puzzle** and lining all its pieces up in a single straight row:
- **Do you still have all the pieces?** Yes, you have 100% of the pieces (information).
- **Can you see the picture?** No, the picture is completely destroyed because the spatial relations between the pieces are gone.

This is exactly what happens when we use a **Flatten Layer** to feed an image into a **Dense (Fully Connected) Layer**. It destroys the grid layout, forcing the network to learn which inputs are related from scratch.""",
        'key_concept': "Dense layers treat each input feature independently. They lack any prior bias or understanding that two adjacent pixels in the input space are related.",
        'sec2_header': "2. Loss of Position Information",
        'col1_subheader': "Before Flattening (2D Image)",
        'col2_subheader': "After Flattening (1D Vector)",
        'observe_grid': """**Observe the Grid:**
* **Cell A** (red, value 1) is at `(1, 2)`.
* **Cell B** (cyan, value 2) is at `(2, 2)`.
* They are direct **vertical neighbors** sharing a horizontal border.""",
        'btn_flatten': "Apply Flatten Layer ➡️",
        'btn_reset': "Reset View 🔄",
        'ready_to_flatten_title': "Ready to Flatten",
        'ready_to_flatten_desc': 'Click the "Apply Flatten Layer" button in the left column to run row-major matrix flattening.',
        'spatial_analysis': "🔍 Spatial Distance Analysis",
        'metric_2d_label': "2D GRID DISTANCE (A to B)",
        'metric_2d_sub': "Adjacent vertical neighbors",
        'metric_1d_label': "1D VECTOR DISTANCE (A to B)",
        'metric_1d_sub': "Separated by 3 elements!",
        'math_title': "💥 The Math Behind the Loss of Proximity",
        'math_body': """When an image is flattened, we map a 2D position $(r, c)$ to a 1D index $i$ using row-major ordering:

$$\\text{Index } i = r \\times \\text{Width} + c$$

Let's compute the indices for our highlighted cells:
- **Cell A** at $(1, 2)$ becomes: $1 \\times 4 + 2 = \\mathbf{6}$
- **Cell B** at $(2, 2)$ becomes: $2 \\times 4 + 2 = \\mathbf{10}$

In 2D, they were direct neighbors sharing an edge. In 1D, they are separated by **4 steps**!""",
        'callout_title': "⚠️ Scaling up to Real-world Images",
        'callout_body': "If this were a modest <b>256 × 256 pixel image</b>, two vertical neighbors would be separated by <b>256 steps</b> in the flattened vector. The dense layer has no spatial prior; it doesn't know that index 6 and index 10 (or index k and index k + 256) are physically adjacent. It has to learn all connections from scratch, destroying rotational and translational invariance!",
        'hover_2d_a': "Cell A (Row {r}, Col {c})<br>Value: {val} (Highlight A)",
        'hover_2d_b': "Cell B (Row {r}, Col {c})<br>Value: {val} (Highlight B)",
        'hover_2d_shape': "Shape Element (Row {r}, Col {c})<br>Value: {val}",
        'hover_2d_bg': "Background (Row {r}, Col {c})<br>Value: {val}",
        'hover_1d_a': "Index {i} (from Row {r}, Col {c})<br>Value: {val} (Highlight A)",
        'hover_1d_b': "Index {i} (from Row {r}, Col {c})<br>Value: {val} (Highlight B)",
        'hover_1d_shape': "Index {i} (from Row {r}, Col {c})<br>Value: {val} (Shape Element)",
        'hover_1d_bg': "Index {i} (from Row {r}, Col {c})<br>Value: {val} (Background)",
        'col_titles_2d': ['Col 0', 'Col 1', 'Col 2', 'Col 3'],
        'row_titles_2d': ['Row 0', 'Row 1', 'Row 2', 'Row 3'],
        'col_title_1d': 'Vector',
        'row_titles_1d_prefix': 'Index',
        'label_shape': 'Shape',
        'input_source': 'Image Source',
        'src_default': 'Default Mock Image',
        'src_custom': 'Custom 4x4 Grid',
        'src_upload': 'Upload Image',
        'upload_label': 'Upload an image (PNG, JPG)'
    },
    'KOR': {
        'title': "인접(Dense/Flatten) 레이어의 한계",
        'subtitle': "신경망에서 공간 정보 손실을 시각적으로 이해하는 인터랙티브 가이드",
        'sec1_header': "1. 공간적 관계의 보존",
        'sec1_p1': """시각적 작업(이미지 분류나 객체 탐지 등)에서 **이미지는 단순한 픽셀 색상의 무작위 모음이 아니며, 고유한 구조를 가집니다.**
물리적으로 서로 가까운 픽셀들은 경선(edges), 질감(textures), 형태(shapes) 및 복잡한 특징을 형성합니다. 예를 들어, 사람의 눈, 코, 입을 나타내는 픽셀들은 항상 특정 공간적 이웃 내에 함께 모여 있습니다.""",
        'analogy_title': "🧩 직소 퍼즐 비유",
        'analogy_body': """완성된 **직소 퍼즐**을 가져와서 모든 조각을 일렬로 길게 늘어놓는다고 상상해보세요:
- **조각이 모두 남아있나요?** 네, 100%의 조각(정보)이 존재합니다.
- **그림을 볼 수 있나요?** 아뇨, 조각들 사이의 공간적 관계가 완전히 파괴되었기 때문에 그림을 볼 수 없습니다.

이것이 바로 이미지를 **인접(Dense/Fully Connected) 레이어**에 입력하기 위해 **Flatten 레이어**를 사용할 때 일어나는 현상입니다. 2D 격자 구조를 파괴하여 네트워크가 어떤 입력들이 서로 관련되어 있는지 처음부터 새로 학습해야 하도록 만듭니다.""",
        'key_concept': "인접 레이어는 각 입력 특징을 독립적으로 취급합니다. 입력 공간에서 인접한 두 픽셀이 관련되어 있다는 사전 편향이나 지식이 없습니다.",
        'sec2_header': "2. 위치 정보의 손실",
        'col1_subheader': "Flatten 전 (2D 이미지)",
        'col2_subheader': "Flatten 후 (1D 벡터)",
        'observe_grid': """**격자 관찰:**
* **셀 A** (빨간색, 값 1)는 `(1, 2)`에 위치합니다.
* **셀 B** (하늘색, 값 2)는 `(2, 2)`에 위치합니다.
* 두 셀은 가로 경계선을 공유하는 직속 **세로 이웃**입니다.""",
        'btn_flatten': "Flatten 레이어 적용 ➡️",
        'btn_reset': "화면 초기화 🔄",
        'ready_to_flatten_title': "Flatten 준비 완료",
        'ready_to_flatten_desc': '왼쪽 열의 "Flatten 레이어 적용" 버튼을 클릭하여 행 우선(row-major) 행렬 Flatten을 시각화하세요.',
        'spatial_analysis': "🔍 공간적 거리 분석",
        'metric_2d_label': "2D 격자 거리 (A에서 B까지)",
        'metric_2d_sub': "인접한 세로 이웃",
        'metric_1d_label': "1D 벡터 거리 (A에서 B까지)",
        'metric_1d_sub': "3개의 원소만큼 떨어짐!",
        'math_title': "💥 인접성 상실의 수학적 원리",
        'math_body': """이미지를 Flatten(평탄화)할 때, 행 우선 순서(row-major ordering)를 사용하여 2D 위치 $(r, c)$를 1D 인덱스 $i$로 매핑합니다:

$$\\text{인덱스 } i = r \\times \\text{가로폭} + c$$

강조된 셀들의 인덱스를 계산해 봅시다:
- **셀 A** $(1, 2)$의 변환: $1 \\times 4 + 2 = \\mathbf{6}$
- **셀 B** $(2, 2)$의 변환: $2 \\times 4 + 2 = \\mathbf{10}$

2D 상에서는 모서리를 맞대고 있는 이웃이었으나, 1D 상에서는 **4단계**나 멀어지게 됩니다!""",
        'callout_title': "⚠️ 실제 이미지 크기에서의 확장성",
        'callout_body': "만약 이것이 평범한 <b>256 × 256 픽셀 이미지</b>였다면, 두 세로 이웃은 Flatten된 벡터에서 무려 <b>256단계</b>나 떨어지게 됩니다. 인접 레이어는 공간적 사전 정보가 없기 때문에 인덱스 6과 인덱스 10(또는 인덱스 k와 k+256)이 물리적으로 인접하다는 사실을 모릅니다. 따라서 모든 관계를 데이터로부터 처음부터 새로 학습해야 하므로, 회전 및 병진 불변성(rotational & translational invariance)이 손실됩니다!",
        'hover_2d_a': "셀 A (행 {r}, 열 {c})<br>값: {val} (강조 A)",
        'hover_2d_b': "셀 B (행 {r}, 열 {c})<br>값: {val} (강조 B)",
        'hover_2d_shape': "도형 부분 (행 {r}, 열 {c})<br>값: {val}",
        'hover_2d_bg': "배경 (행 {r}, 열 {c})<br>값: {val}",
        'hover_1d_a': "인덱스 {i} (행 {r}, 열 {c}에서 변환)<br>값: {val} (강조 A)",
        'hover_1d_b': "인덱스 {i} (행 {r}, 열 {c}에서 변환)<br>값: {val} (강조 B)",
        'hover_1d_shape': "인덱스 {i} (행 {r}, 열 {c}에서 변환)<br>값: {val}",
        'hover_1d_bg': "인덱스 {i} (행 {r}, 열 {c}에서 변환)<br>값: {val}",
        'col_titles_2d': ['열 0', '열 1', '열 2', '열 3'],
        'row_titles_2d': ['행 0', '행 1', '행 2', '행 3'],
        'col_title_1d': '벡터',
        'row_titles_1d_prefix': '인덱스',
        'label_shape': '도형',
        'input_source': '이미지 소스',
        'src_default': '기본 모의 이미지',
        'src_custom': '사용자 지정 4x4 그리드',
        'src_upload': '이미지 업로드',
        'upload_label': '이미지 업로드 (PNG, JPG)'
    }
}

# ==============================================================================
# HEADER BANNER & LANGUAGE TOGGLE
# ==============================================================================
# Dynamic header utilizing st.columns to place the language toggle at the top right
header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.markdown(f'<div class="main-title">{t[st.session_state.lang]["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{t[st.session_state.lang]["subtitle"]}</div>', unsafe_allow_html=True)

with header_col2:
    # Add vertical spacing to align buttons with the title text
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    lang_col2, lang_col1 = st.columns(2)
    with lang_col1:
        if st.button("ENG", type="primary" if st.session_state.lang == 'ENG' else "secondary", use_container_width=True):
            st.session_state.lang = 'ENG'
            st.rerun()
    with lang_col2:
        if st.button("KOR", type="primary" if st.session_state.lang == 'KOR' else "secondary", use_container_width=True):
            st.session_state.lang = 'KOR'
            st.rerun()

# ==============================================================================
# SECTION 1: PRESERVATION OF SPATIAL RELATIONSHIPS
# ==============================================================================
st.header(t[st.session_state.lang]['sec1_header'])

st.markdown(t[st.session_state.lang]['sec1_p1'])

st.markdown(f"### {t[st.session_state.lang]['analogy_title']}")
st.markdown(t[st.session_state.lang]['analogy_body'])

st.info(f"💡 **Key Concept:** {t[st.session_state.lang]['key_concept']}", icon="🧠")

st.markdown("---")

# ==============================================================================
# SECTION 2: LOSS OF POSITION INFORMATION (INTERACTIVE VISUALIZATION)
# ==============================================================================
st.header(t[st.session_state.lang]['sec2_header'])

colorscale = [
    [0.0, '#1E1E2F'],   # 0 -> Background
    [0.166, '#1E1E2F'],
    [0.166, '#FF4B4B'], # 1 -> Cell A (Red)
    [0.5, '#FF4B4B'],
    [0.5, '#00F5FF'],   # 2 -> Cell B (Cyan)
    [0.833, '#00F5FF'],
    [0.833, '#AB63FA'], # 3 -> Shape Element (Purple)
    [1.0, '#AB63FA']
]

st.markdown(f"### {t[st.session_state.lang]['input_source']}")
input_type = st.radio("Image Source", [
    t[st.session_state.lang]['src_default'],
    t[st.session_state.lang]['src_custom'],
    t[st.session_state.lang]['src_upload']
], horizontal=True, label_visibility="collapsed")

if input_type == t[st.session_state.lang]['src_custom']:
    default_grid = [
        [0, 0, 3, 0],
        [0, 3, 1, 3],
        [0, 0, 2, 0],
        [0, 0, 3, 0]
    ]
    edited_df = st.data_editor(pd.DataFrame(default_grid), use_container_width=False, hide_index=True)
    grid = edited_df.to_numpy()
    colorscale_used = colorscale
elif input_type == t[st.session_state.lang]['src_upload']:
    uploaded_file = st.file_uploader(t[st.session_state.lang]['upload_label'], type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        img = img.resize((64, 64))
        grid = np.array(img)
    else:
        grid = np.zeros((4, 4))
    colorscale_used = 'gray'
else:
    grid = np.array([
        [0, 0, 3, 0],
        [0, 3, 1, 3],
        [0, 0, 2, 0],
        [0, 0, 3, 0]
    ])
    colorscale_used = colorscale

# Side-by-side columns comparison
col1, col2 = st.columns([1, 1])

rows, cols = grid.shape

# Generate bilingual hover text for the 2D grid
hover_text_2d = []
for r in range(rows):
    row_text = []
    for c in range(cols):
        val = grid[r, c]
        if r == 1 and c == 2 and rows == 4:
            row_text.append(t[st.session_state.lang]['hover_2d_a'].format(r=r, c=c, val=val))
        elif r == 2 and c == 2 and rows == 4:
            row_text.append(t[st.session_state.lang]['hover_2d_b'].format(r=r, c=c, val=val))
        elif val == 3 and input_type != t[st.session_state.lang]['src_upload'] and rows == 4:
            row_text.append(t[st.session_state.lang]['hover_2d_shape'].format(r=r, c=c, val=val))
        else:
            row_text.append(t[st.session_state.lang]['hover_2d_bg'].format(r=r, c=c, val=val))
    hover_text_2d.append(row_text)

# Build 2D Heatmap Fig
fig_2d = go.Figure(data=go.Heatmap(
    z=grid,
    colorscale=colorscale_used,
    showscale=False,
    xgap=5 if cols <= 4 else 0,
    ygap=5 if rows <= 4 else 0,
    hoverinfo='text',
    text=hover_text_2d
))

# Configure 2D Plot
fig_2d.update_layout(
    title=dict(text=t[st.session_state.lang]['col1_subheader'], font=dict(size=18, family="Outfit")),
    xaxis=dict(
        tickmode='array' if cols <= 4 else 'auto', 
        tickvals=[0, 1, 2, 3] if cols <= 4 else None, 
        ticktext=t[st.session_state.lang]['col_titles_2d'] if cols <= 4 else None, 
        side='top',
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        tickmode='array' if rows <= 4 else 'auto', 
        tickvals=[0, 1, 2, 3] if rows <= 4 else None, 
        ticktext=t[st.session_state.lang]['row_titles_2d'] if rows <= 4 else None, 
        autorange='reversed',
        showgrid=False,
        zeroline=False
    ),
    width=400,
    height=400,
    margin=dict(l=40, r=40, t=65, b=40),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

# Text annotation for grid cells
if rows <= 4:
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            if r == 1 and c == 2:
                label = "A"
                color = "white" if input_type != t[st.session_state.lang]['src_upload'] else "red"
            elif r == 2 and c == 2:
                label = "B"
                color = "#1E1E2F" if input_type != t[st.session_state.lang]['src_upload'] else "cyan"
            elif val == 3 and input_type != t[st.session_state.lang]['src_upload']:
                label = t[st.session_state.lang]['label_shape']
                color = "white"
            else:
                label = ""
                color = "gray"
                
            fig_2d.add_annotation(
                x=c,
                y=r,
                text=label,
                showarrow=False,
                font=dict(color=color, size=13, family="Outfit", weight="bold"),
                xanchor='center',
                yanchor='middle'
            )

# Render Left Column (2D Grid)
with col1:
    st.plotly_chart(fig_2d, use_container_width=True)
    st.markdown(t[st.session_state.lang]['observe_grid'])
    
    if not st.session_state.flattened:
        if st.button(t[st.session_state.lang]['btn_flatten'], type="primary", use_container_width=True):
            st.session_state.flattened = True
            st.rerun()

# Render Right Column (1D Vector)
with col2:
    if st.session_state.flattened:
        vector_len = rows * cols
        vector = grid.flatten().reshape(vector_len, 1)
        
        # Hover text for the 1D vector
        hover_text_1d = []
        for i in range(vector_len):
            val = vector[i, 0]
            r = i // cols
            c = i % cols
            if r == 1 and c == 2 and rows == 4:
                text = t[st.session_state.lang]['hover_1d_a'].format(i=i, r=r, c=c, val=val)
            elif r == 2 and c == 2 and rows == 4:
                text = t[st.session_state.lang]['hover_1d_b'].format(i=i, r=r, c=c, val=val)
            elif val == 3 and input_type != t[st.session_state.lang]['src_upload'] and rows == 4:
                text = t[st.session_state.lang]['hover_1d_shape'].format(i=i, r=r, c=c, val=val)
            else:
                text = t[st.session_state.lang]['hover_1d_bg'].format(i=i, r=r, c=c, val=val)
            hover_text_1d.append([text])
            
        # Build 1D Plot
        fig_1d = go.Figure(data=go.Heatmap(
            z=vector,
            x=[t[st.session_state.lang]['col_title_1d']],
            y=list(range(vector_len)),
            colorscale=colorscale_used,
            showscale=False,
            xgap=5 if cols <= 4 else 0,
            ygap=5 if vector_len <= 16 else 0,
            hoverinfo='text',
            text=hover_text_1d
        ))
        
        fig_1d.update_layout(
            title=dict(text=t[st.session_state.lang]['col2_subheader'], font=dict(size=18, family="Outfit")),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(
                tickmode='array' if vector_len <= 16 else 'auto', 
                tickvals=list(range(vector_len)) if vector_len <= 16 else None, 
                ticktext=[f"{t[st.session_state.lang]['row_titles_1d_prefix']} {i}" for i in range(vector_len)] if vector_len <= 16 else None, 
                autorange='reversed',
                showgrid=False,
                zeroline=False
            ),
            width=250,
            height=600,
            margin=dict(l=60, r=40, t=65, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Add labels to the vector cells
        if vector_len <= 16:
            for i in range(vector_len):
                val = vector[i, 0]
                r = i // cols
                c = i % cols
                if r == 1 and c == 2:
                    label = f"A ({t[st.session_state.lang]['row_titles_1d_prefix']} {i})"
                    color = "white" if input_type != t[st.session_state.lang]['src_upload'] else "red"
                    size = 12
                    weight = "bold"
                elif r == 2 and c == 2:
                    label = f"B ({t[st.session_state.lang]['row_titles_1d_prefix']} {i})"
                    color = "#1E1E2F" if input_type != t[st.session_state.lang]['src_upload'] else "cyan"
                    size = 12
                    weight = "bold"
                elif val == 3 and input_type != t[st.session_state.lang]['src_upload']:
                    label = f"{t[st.session_state.lang]['label_shape']} ({i})"
                    color = "white"
                    size = 10
                    weight = "normal"
                else:
                    label = f"{t[st.session_state.lang]['row_titles_1d_prefix']} {i}"
                    color = "#666688"
                    size = 10
                    weight = "normal"
                    
                fig_1d.add_annotation(
                    x=0,
                    y=i,
                    text=label,
                    showarrow=False,
                    font=dict(
                        color=color, 
                        size=size, 
                        family="Outfit", 
                        weight=weight
                    ),
                    xanchor='center',
                    yanchor='middle'
                )
            
        st.plotly_chart(fig_1d, use_container_width=True)
        
        if st.button(t[st.session_state.lang]['btn_reset'], use_container_width=True):
            st.session_state.flattened = False
            st.rerun()
            
    else:
        # Styled placeholder info when not yet flattened
        st.markdown(
            f"""
            <div style="border: 2px dashed #3A3A5A; border-radius: 12px; height: 400px; display: flex; align-items: center; justify-content: center; text-align: center; padding: 2rem; background-color: #161625;">
                <div>
                    <span style="font-size: 3.5rem;">⚙️</span>
                    <h3 style="margin-top: 1rem; color: #FFFFFF;">{t[st.session_state.lang]['ready_to_flatten_title']}</h3>
                    <p style="color: #8888AA; max-width: 300px; margin: 0.5rem auto;">{t[st.session_state.lang]['ready_to_flatten_desc']}</p>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

# ==============================================================================
# ANALYSIS AND EXPLANATION SECTION
# ==============================================================================
if st.session_state.flattened:
    st.markdown("---")
    st.subheader(t[st.session_state.lang]['spatial_analysis'])
    
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="color: #8888AA; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.05em;">{t[st.session_state.lang]['metric_2d_label']}</div>
                <div class="metric-val-2d">1.0</div>
                <div style="color: #FF4B4B; font-size: 0.85rem; font-weight: 600;">{t[st.session_state.lang]['metric_2d_sub']}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col_metric2:
        metric_1d_sub_text = t[st.session_state.lang]['metric_1d_sub']
        if grid.shape[1] != 4:
            metric_1d_sub_text = metric_1d_sub_text.replace('3', str(grid.shape[1]-1))
            
        st.markdown(
            f"""
            <div class="metric-container">
                <div style="color: #8888AA; font-size: 0.95rem; font-weight: 600; letter-spacing: 0.05em;">{t[st.session_state.lang]['metric_1d_label']}</div>
                <div class="metric-val-1d">{float(grid.shape[1])}</div>
                <div style="color: #00F5FF; font-size: 0.85rem; font-weight: 600;">{metric_1d_sub_text}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    math_body_text = t[st.session_state.lang]['math_body']
    if grid.shape[1] != 4:
        math_body_text = math_body_text.replace(' 4 ', f' {grid.shape[1]} ')
        math_body_text = math_body_text.replace('4 steps', f'{grid.shape[1]} steps')
        math_body_text = math_body_text.replace('4단계', f'{grid.shape[1]}단계')
        math_body_text = math_body_text.replace('6', str(1 * grid.shape[1] + 2))
        math_body_text = math_body_text.replace('10', str(2 * grid.shape[1] + 2))

    st.markdown(f"### {t[st.session_state.lang]['math_title']}")
    st.markdown(math_body_text)
    
    st.markdown(
        f"""
        <div class="callout-card">
            <h4 style="margin-top:0; color: #AB63FA;">{t[st.session_state.lang]['callout_title']}</h4>
            {t[st.session_state.lang]['callout_body']}
        </div>
        """,
        unsafe_allow_html=True
    )
