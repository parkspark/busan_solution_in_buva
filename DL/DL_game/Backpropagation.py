import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os

# TensorFlow 로그 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.datasets import fashion_mnist

st.set_page_config(page_title="Backprop Hero", layout="wide")

# Custom CSS for UI
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
    }
    .hero-title {
        color: #ff0055;
        text-shadow: 0 0 10px rgba(255,0,85,0.8);
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Initialize Session State
if 'initialized' not in st.session_state:
    (x_train, y_train), _ = fashion_mnist.load_data()
    idx = 0 # Ankle boot 샘플
    img = x_train[idx] / 255.0
    label = y_train[idx]
    
    st.session_state.x = img.reshape(784, 1)
    st.session_state.label = label
    st.session_state.y_true = np.zeros((10, 1))
    st.session_state.y_true[label] = 1.0
    
    # He Initialization (ReLU용)
    np.random.seed(42)
    st.session_state.W1 = np.random.randn(64, 784) * np.sqrt(2. / 784)
    st.session_state.b1 = np.zeros((64, 1))
    st.session_state.W2 = np.random.randn(10, 64) * np.sqrt(2. / 64)
    st.session_state.b2 = np.zeros((10, 1))
    
    st.session_state.epoch = 0
    st.session_state.loss_history = []
    st.session_state.state = 'start' # 상태: start, forwarded, backwarded
    
    st.session_state.z1 = None
    st.session_state.a1 = None
    st.session_state.a2 = None
    st.session_state.loss = None
    st.session_state.initialized = True
    
def reset_game():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# 2. Sidebar Layout
st.sidebar.title("🛠️ 컨트롤 패널")
st.session_state.lr = st.sidebar.slider("학습률 (Learning Rate)", 0.001, 1.0, 0.1)
st.sidebar.metric("현재 Epoch", f"{st.session_state.epoch} / 20")
st.sidebar.metric("목표 Loss", "<= 0.1")

if st.sidebar.button("🔄 게임 초기화"):
    reset_game()
    st.rerun()

# 3. Main Header
st.markdown('<h1 class="hero-title">🦸 Backprop Hero: 역전파를 마스터하라!</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1em;'>단 하나의 이미지를 신경망이 완벽하게 맞추도록(학습하도록) 만들어보세요!<br><b>▶ 순전파</b>로 오차를 구하고, <b>◀ 역전파</b>로 오차를 거슬러 올라가며 가중치를 수정합니다.</p>", unsafe_allow_html=True)

# 4. Action Buttons (Game Flow)
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    cc1, cc2 = st.columns(2)
    with cc1:
        if st.button("▶ 1단계: 순전파 (Forward)"):
            if st.session_state.epoch >= 20:
                st.warning("에포크 제한(20)에 도달했습니다. 게임을 초기화해주세요.")
            else:
                # Numpy 수동 순전파 연산
                z1 = np.dot(st.session_state.W1, st.session_state.x) + st.session_state.b1
                a1 = np.maximum(0, z1) # ReLU 활성화
                z2 = np.dot(st.session_state.W2, a1) + st.session_state.b2
                # Softmax 출력
                exp_z2 = np.exp(z2 - np.max(z2))
                a2 = exp_z2 / np.sum(exp_z2)
                
                # Cross-Entropy Loss
                loss = -np.sum(st.session_state.y_true * np.log(a2 + 1e-8))
                
                st.session_state.z1 = z1
                st.session_state.a1 = a1
                st.session_state.a2 = a2
                st.session_state.loss = loss
                st.session_state.state = 'forwarded'
                
    with cc2:
        if st.button("◀ 2단계: 역전파 (Backward)"):
            if st.session_state.state != 'forwarded':
                st.error("먼저 순전파를 실행하여 현재 오차(Loss)를 구해야 역전파가 가능합니다!")
            elif st.session_state.epoch >= 20:
                st.warning("에포크 제한(20)에 도달했습니다.")
            else:
                # Numpy 수동 역전파(기울기) 연산 (Chain Rule)
                dz2 = st.session_state.a2 - st.session_state.y_true
                dW2 = np.dot(dz2, st.session_state.a1.T)
                db2 = dz2
                
                da1 = np.dot(st.session_state.W2.T, dz2)
                dz1 = da1 * (st.session_state.z1 > 0) # ReLU 미분
                dW1 = np.dot(dz1, st.session_state.x.T)
                db1 = dz1
                
                # 가중치 업데이트 (SGD)
                lr = st.session_state.lr
                st.session_state.W1 -= lr * dW1
                st.session_state.b1 -= lr * db1
                st.session_state.W2 -= lr * dW2
                st.session_state.b2 -= lr * db2
                
                st.session_state.epoch += 1
                st.session_state.loss_history.append(st.session_state.loss)
                st.session_state.state = 'backwarded'

# 5. Visualizations
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. 입력 (Input Data)")
    img_2d = st.session_state.x.reshape(28, 28)
    fig1 = go.Figure(data=go.Heatmap(z=img_2d, colorscale='gray', showscale=False))
    fig1.update_layout(yaxis=dict(autorange='reversed'), height=350, margin=dict(l=10, r=10, b=10, t=10), template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)
    st.caption(f"Fashion MNIST 정답 레이블: **{st.session_state.label} (Ankle boot)**")

with col2:
    st.subheader("2. 은닉층 가중치 (W2 Matrix)")
    st.caption("클래스(10) x 은닉노드(64). 역전파 시 색상이 업데이트됩니다.")
    fig2 = go.Figure(data=go.Heatmap(z=st.session_state.W2, colorscale='RdBu', zmin=-0.5, zmax=0.5))
    fig2.update_layout(height=350, margin=dict(l=10, r=10, b=10, t=10), template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    st.subheader("3. 예측 및 오차 (Prediction)")
    if st.session_state.state in ['forwarded', 'backwarded']:
        preds = st.session_state.a2.flatten()
        colors = ['#00ffcc'] * 10
        colors[st.session_state.label] = '#ff0055' # 정답 클래스 강렬하게 표시
        
        fig3 = go.Figure(data=go.Bar(x=[str(i) for i in range(10)], y=preds, marker_color=colors))
        fig3.update_layout(title=f"<span style='color:#ff0055'>현재 Loss: {st.session_state.loss:.4f}</span>", 
                           yaxis=dict(range=[0, 1]), height=350, margin=dict(l=10, r=10, b=10, t=40), template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("순전파를 실행하여 모델의 첫 예측값을 확인하세요!")

st.markdown("---")
# Loss Curve Race
if len(st.session_state.loss_history) > 0:
    fig_loss = go.Figure(data=go.Scatter(y=st.session_state.loss_history, mode='lines+markers', line=dict(color='#ff0055', width=3), marker=dict(size=8)))
    fig_loss.update_layout(title="학습 오차 곡선 (Loss Curve)", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_loss, use_container_width=True)

# 6. Win / Lose Conditions
if st.session_state.loss is not None and st.session_state.loss <= 0.1:
    st.balloons()
    st.success(f"🎉 **미션 클리어!** 단 {st.session_state.epoch} 에포크 만에 목표 Loss(0.1 이하)를 달성했습니다! 역전파 알고리즘이 정답을 향해 완벽하게 가중치를 수정해냈습니다.")
elif st.session_state.epoch >= 20:
    st.error("💀 에포크 제한(20)에 도달했습니다. 학습률(LR)이 너무 작거나 커서 수렴하지 못했습니다. 왼쪽에서 게임 초기화 후 다시 도전하세요!")
