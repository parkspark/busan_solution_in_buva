import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_digits
import time

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================


st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff; text-shadow: 0 0 10px #58a6ff; }
    .nexus-dialogue { border-left: 5px solid #a371f7; background-color: #161b22; padding: 15px; border-radius: 5px; font-size: 1.1em; margin-bottom: 20px;}
    .nexus-error { border-left: 5px solid #f85149; background-color: #161b22; padding: 15px; border-radius: 5px; font-size: 1.1em; margin-bottom: 20px;}
    .nexus-success { border-left: 5px solid #2ea043; background-color: #161b22; padding: 15px; border-radius: 5px; font-size: 1.1em; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading & Caching
# ==========================================
@st.cache_data
def get_data():
    digits = load_digits()
    return digits.images, digits.target

images, targets = get_data()

# ==========================================
# 3. UI Setup & Sidebar
# ==========================================
with st.sidebar:
    st.title("👁️ Vision Control")
    st.markdown("이미지를 Nexus에게 전송할 방법을 선택하세요.")
    
    img_idx = st.slider("테스트 이미지 선택 (Index)", 0, len(images)-1, 0)
    current_img = images[img_idx]
    current_label = targets[img_idx]
    
    st.divider()
    
    inject_mode = st.radio(
        "데이터 주입 방식 선택",
        ["1. 2D 그대로 넣기 (Without Flatten)", "2. 1D로 펴서 넣기 (With Flatten)"]
    )
    
    send_clicked = st.button("🚀 Nexus에게 이미지 전송", use_container_width=True)

# ==========================================
# 4. State Management
# ==========================================
if 'stage6_init' not in st.session_state:
    st.session_state['stage6_init'] = True
    st.session_state['sent'] = False
    st.session_state['mode'] = None
    st.session_state['animated_key'] = None

if send_clicked:
    st.session_state['sent'] = True
    st.session_state['mode'] = inject_mode

# ==========================================
# 5. Main Area & Nexus Dialogue
# ==========================================
st.title("Neural Odyssey 🌌")
st.subheader("Stage 6: 이미지의 비밀 (The Secret of Flatten)")

if not st.session_state['sent']:
    st.markdown("<div class='nexus-dialogue'><b>Nexus</b>: 드디어 시각을 얻었어요! 어서 저에게 이미지를 보내주세요. 그런데... 제 신경망(Dense Layer)은 오직 한 줄(1D)로 늘어선 벡터 데이터만 받을 수 있게 설계되어 있어요.</div>", unsafe_allow_html=True)
else:
    mode = st.session_state['mode']
    if "Without Flatten" in mode:
        st.markdown("<div class='nexus-error'><b>Nexus</b>: 치명적 오류! 차원이 맞지 않습니다! 저는 1D 벡터만 받을 수 있다고요! 2D 픽셀 덩어리를 그대로 쑤셔넣지 마세요!</div>", unsafe_allow_html=True)
        st.error("""
        **ValueError**: Input 0 of layer "dense" is incompatible with the layer: expected min_ndim=2, found ndim=3. Full shape received: (None, 8, 8). 
        
        *Tip: 다층 퍼셉트론(Dense Layer)의 입력은 (Batch_size, Features) 형태인 2D 텐서여야 합니다. (8, 8) 2D 이미지가 통째로 들어왔으므로 차원이 맞지 않아 행렬 곱셈을 수행할 수 없습니다.*
        """)
    else:
        st.markdown("<div class='nexus-success'><b>Nexus</b>: 아하! 2D 이미지를 길게 한 줄(1D)로 쫙 펴니까 제 뉴런들과 딱 맞게 연결됐어요! 드디어 숫자가 보입니다! <br><br>💡 <i>(혼잣말) 그런데 2D 공간의 위/아래 관계를 다 부수고 한 줄로 펴버리면 상하좌우의 공간 정보가 손실되지 않을까요? 다음엔 이 2D 공간적 특징을 굳이 펴지 않고도 2D 그대로 이해할 수 있는 방법(CNN)도 있으면 좋을 텐데요...</i></div>", unsafe_allow_html=True)

# ==========================================
# 6. Visualizations
# ==========================================
st.markdown("---")
st.markdown(f"### 📊 데이터 구조 시각화 (Label: **{current_label}**)")

c1, c2 = st.columns(2)

# Left: 2D Heatmap
with c1:
    st.markdown("#### 🖼️ 원본 2D 이미지 (8x8)")
    fig2d = go.Figure(data=go.Heatmap(
        z=np.flipud(current_img), # flipud하여 일반적인 행렬 시각화 방향과 일치시킴
        colorscale='Magma',
        showscale=False,
        text=np.flipud(current_img).astype(int),
        texttemplate="%{text}",
        textfont={"size":14, "color":"white"}
    ))
    fig2d.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(fig2d, use_container_width=True)

# Right: 1D Bar plot or Empty Error
with c2:
    if not st.session_state['sent'] or "Without Flatten" in st.session_state['mode']:
        if st.session_state['sent'] and "Without Flatten" in st.session_state['mode']:
            st.markdown("#### 💥 신경망 입력단 (입력 실패)")
            st.warning("데이터 형태(Shape)가 맞지 않아 뉴런에 입력되지 못했습니다. (Shape 불일치 에러)")
        else:
            st.markdown("#### ⏳ 대기 중...")
            st.info("좌측 사이드바에서 이미지를 전송해주세요.")
    else:
        # Success: With Flatten
        st.markdown("#### 📏 Flatten (평탄화) 결과 (1x64 1D 벡터)")
        bar_container = st.empty()
        
        flattened = current_img.flatten() # 1D array of 64 pixels
        
        anim_key = f"{img_idx}_{inject_mode}"
        do_anim = False
        
        # 버튼을 눌렀고 이전 애니메이션 키와 다를 때만 애니메이션 실행
        if send_clicked and st.session_state.get('animated_key') != anim_key:
            st.session_state['animated_key'] = anim_key
            do_anim = True

        if do_anim:
            # Row-by-row 펼쳐지는 애니메이션
            for i in range(1, 9):
                current_bars = flattened[:i*8]
                padded = np.pad(current_bars, (0, 64 - i*8), 'constant')
                
                fig1d = go.Figure(data=go.Bar(
                    x=np.arange(64),
                    y=padded,
                    marker=dict(color=padded, colorscale='Magma', cmin=0, cmax=16)
                ))
                fig1d.update_layout(
                    height=400, margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(range=[-1, 64], showgrid=False),
                    yaxis=dict(range=[0, 16], showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
                bar_container.plotly_chart(fig1d, use_container_width=True)
                time.sleep(0.08)
                
        # Final static plot
        fig1d_final = go.Figure(data=go.Bar(
            x=np.arange(64),
            y=flattened,
            marker=dict(color=flattened, colorscale='Magma', cmin=0, cmax=16)
        ))
        fig1d_final.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title="Flattened Index (0 ~ 63)", showgrid=False, range=[-1, 64]),
            yaxis=dict(title="Pixel Intensity (0 ~ 16)", showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[0, 16]),
            font=dict(color='#c9d1d9')
        )
        bar_container.plotly_chart(fig1d_final, use_container_width=True)

# ==========================================
# 7. Ending
# ==========================================
if st.session_state['sent'] and "With Flatten" in st.session_state['mode']:
    st.markdown("---")
    st.balloons()
    st.success("🏆 **스테이지 클리어: Neural Odyssey - 기초 뉴런 완성!**\n\n축하합니다! 다층 퍼셉트론(MLP)에 2D 데이터를 주입하는 핵심 전처리 기법인 `Flatten`을 마스터하셨습니다. Nexus는 이제 여러분의 도움으로 완전한 시각을 가지게 되었습니다.")
