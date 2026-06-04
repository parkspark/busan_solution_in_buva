import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import os

# TensorFlow 로그 최소화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 기본 설정 ---
st.set_page_config(page_title="Neural Odyssey: Stage 9", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS (Neon Dark Theme) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
    }
    h1, h2, h3 {
        color: #00ffcc;
        text-shadow: 0 0 8px rgba(0, 255, 204, 0.5);
    }
    .nexus-dialogue {
        background: linear-gradient(90deg, rgba(0, 255, 204, 0.1) 0%, rgba(255, 0, 255, 0.05) 100%);
        border-left: 5px solid #00ffcc;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(0, 255, 204, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 유틸리티 및 데이터 셋업 ---
@st.cache_data
def get_data():
    # 8x8 픽셀의 숫자 이미지 데이터 (총 1797개, 64차원)
    digits = load_digits()
    X = digits.data
    y = digits.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def build_model(activation_fn, num_layers, neurons, lr):
    # 가중치 초기화 설정 (균일한 조건 비교를 위해 시드 고정)
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    inputs = tf.keras.Input(shape=(64,))
    x = inputs
    
    # 가변형 은닉층 추가
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(neurons, activation=activation_fn, kernel_initializer=initializer)(x)
        
    # 출력층 (10개 클래스 다중 분류)
    outputs = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 사이드바 (Activation Playground) ---
st.sidebar.title("Activation Playground")
st.sidebar.subheader("🔬 함수 선택 연구소")
explore_act = st.sidebar.selectbox("다양한 활성화 함수를 시각화해보세요:", ["Sigmoid", "Tanh", "ReLU", "LeakyReLU"])

x_val = np.linspace(-5, 5, 100)
if explore_act == "Sigmoid":
    y_val = 1 / (1 + np.exp(-x_val))
    dy_val = y_val * (1 - y_val)
    desc = "미분값이 최대 0.25이므로, 층이 깊어지면 기울기가 소멸(Vanishing Gradient)합니다."
elif explore_act == "Tanh":
    y_val = np.tanh(x_val)
    dy_val = 1 - y_val**2
    desc = "Sigmoid보다 중심이 0으로 맞춰져 학습이 빠르지만, 여전히 양극단에서 기울기가 소멸합니다."
elif explore_act == "ReLU":
    y_val = np.maximum(0, x_val)
    dy_val = np.where(x_val > 0, 1.0, 0.0)
    desc = "양수 영역에서 기울기가 1로 유지되어 소멸을 막습니다! 하지만 음수일 때 기울기가 0이 되어 'Dead ReLU' 위험이 존재합니다."
elif explore_act == "LeakyReLU":
    alpha = 0.1
    y_val = np.where(x_val > 0, x_val, alpha * x_val)
    dy_val = np.where(x_val > 0, 1.0, alpha)
    desc = "음수 영역에서도 미세한 기울기(0.1)를 살려두어 ReLU의 단점인 Dead ReLU 현상을 방지합니다."

fig_sidebar = go.Figure()
fig_sidebar.add_trace(go.Scatter(x=x_val, y=y_val, name="f(x)", line=dict(color="#00ffcc", width=3)))
fig_sidebar.add_trace(go.Scatter(x=x_val, y=dy_val, name="f'(x) (도함수)", line=dict(dash='dash', color="#ff0055", width=2)))
fig_sidebar.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), title=f"{explore_act} 분석", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"))
st.sidebar.plotly_chart(fig_sidebar, use_container_width=True)
st.sidebar.info(desc)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ 아키텍처 세팅")
num_layers = st.sidebar.slider("은닉층(Hidden Layer) 개수", 1, 10, 5)
neurons = st.sidebar.selectbox("레이어 당 뉴런 수", [16, 32, 64, 128], index=1)
learning_rate = st.sidebar.selectbox("학습률 (Learning Rate)", [0.001, 0.01, 0.05, 0.1], index=2, help="학습률이 너무 높으면 과도한 업데이트로 인해 가중치가 음수로 튀면서 Dead ReLU가 대량 발생합니다!")
train_btn = st.sidebar.button("🚀 딥러닝 대결 시작 (Sigmoid vs ReLU)")

st.sidebar.markdown("---")
st.sidebar.subheader("🌟 미션 달성 조건 (별 3개)")
st.sidebar.markdown("- ★ : ReLU + 5층 이상 + 정확도 85% 이상\n- ★★ : Dead ReLU 10% 미만 유지 (안정적인 학습률)\n- ★★★ : 은닉층 뉴런 수 32개 이하 (파라미터 최적화)")

# --- 메인 영역 (Game Area & Revolution Canvas) ---
st.title("Neural Odyssey - Stage 9: ReLU의 각성 (ReLU Revolution)")

nexus_ph = st.empty()
nexus_ph.markdown(f'<div class="nexus-dialogue" style="border-left-color: #ffcc00; color:#ffcc00;">🤖 <b>Nexus:</b> Stage 8에서 깊은 층을 통과하며 잃어버렸던 기울기... 이제 <b>ReLU</b>로 그 한계를 돌파할 수 있을지 시험해볼 시간입니다! 학습 버튼을 눌러주세요.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚔️ Sigmoid vs ReLU 실시간 대결")
    st.markdown("8x8 숫자 이미지(Digits) 다중 분류 시뮬레이션입니다. 층이 깊어질수록 두 함수의 성능 차이가 극명해집니다.")
    plot_ph1 = st.empty()

with col2:
    st.subheader("💀 Dead ReLU 관측기")
    st.markdown("ReLU 신경망의 뉴런 상태입니다. <span style='color:#00ffcc;'>청록색은 살아있는 뉴런</span>, <span style='color:#333344;'>회색은 영원히 0을 출력하는 죽은 뉴런(Dead ReLU)</span>입니다.", unsafe_allow_html=True)
    plot_ph2 = st.empty()

st.markdown("---")
st.subheader("🧩 적재적소 미션 패널 (Output Layer 조율사)")
st.markdown("주어진 문제 상황에 따라 **출력층(Output Layer)**에 가장 알맞은 활성화 함수를 연결하세요. (은닉층에는 보통 ReLU를 씁니다)")
c1, c2, c3 = st.columns(3)
q1 = c1.selectbox("1. 이진 분류 (개 vs 고양이 확률)", ["선택하세요", "Sigmoid", "Softmax", "Linear(없음)"])
q2 = c2.selectbox("2. 다중 분류 (0~9 숫자, 확률합=1)", ["선택하세요", "Sigmoid", "Softmax", "Linear(없음)"])
q3 = c3.selectbox("3. 회귀 (주택 가격 연속값 예측)", ["선택하세요", "Sigmoid", "Softmax", "Linear(없음)"])

mission_passed = (q1 == "Sigmoid" and q2 == "Softmax" and q3 == "Linear(없음)")
if mission_passed:
    st.success("🎉 미션 통과! 활성화 함수의 용도를 완벽히 이해했습니다. (출력층은 목적에 맞게, 은닉층은 ReLU 계열을 쓰는 것이 현대 딥러닝의 정석입니다!)")

if 'cleared' not in st.session_state:
    st.session_state.cleared = False
if 'stars' not in st.session_state:
    st.session_state.stars = 0

# --- 학습 시뮬레이션 로직 ---
if train_btn:
    X_train, X_test, y_train, y_test = get_data()
    
    tf.keras.backend.clear_session()
    model_sig = build_model('sigmoid', num_layers, neurons, learning_rate)
    model_relu = build_model('relu', num_layers, neurons, learning_rate)
    
    # Dead ReLU를 추적하기 위한 Sub-model 구성 (Dense 은닉층의 출력만 가져옴)
    dense_layers = [l for l in model_relu.layers if isinstance(l, tf.keras.layers.Dense)][:-1]
    if dense_layers:
        act_model = tf.keras.Model(inputs=model_relu.input, outputs=[l.output for l in dense_layers])
    else:
        act_model = None
        
    epochs = 30
    acc_sig_history = []
    acc_relu_history = []
    
    progress = st.progress(0)
    
    # Dead ReLU 확인을 위해 데이터 일부만 샘플링 (속도 최적화)
    X_sample = X_train[:256] 
    
    for epoch in range(1, epochs + 1):
        # 1에포크씩 번갈아가며 학습 진행
        hist_sig = model_sig.fit(X_train, y_train, batch_size=128, epochs=1, verbose=0)
        hist_relu = model_relu.fit(X_train, y_train, batch_size=128, epochs=1, verbose=0)
        
        acc_sig = hist_sig.history['accuracy'][0]
        acc_relu = hist_relu.history['accuracy'][0]
        acc_sig_history.append(acc_sig)
        acc_relu_history.append(acc_relu)
        
        # Dead ReLU 계산 로직
        total_dead = 0
        total_neurons_cnt = 0
        z_map = [] # 히트맵 시각화용 데이터
        
        if act_model:
            # 훈련 없이 Forward Pass만 수행하여 활성화 값 추출
            activations = act_model(X_sample, training=False)
            if not isinstance(activations, list):
                activations = [activations]
                
            for act in activations:
                # act shape: (batch_size, num_neurons)
                act_numpy = act.numpy()
                # 미니배치 내의 모든 샘플에 대해 최대 활성화 값이 0 이하라면 해당 뉴런은 "Dead" 상태
                max_act = np.max(act_numpy, axis=0)
                is_dead = (max_act <= 0.0).astype(int) # 1: Dead, 0: Alive
                
                z_map.append(is_dead)
                total_dead += np.sum(is_dead)
                total_neurons_cnt += len(is_dead)
                
        dead_ratio = total_dead / total_neurons_cnt if total_neurons_cnt > 0 else 0.0
        
        # UI 업데이트 1: 정확도 차트 (좌측)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=list(range(1, epoch+1)), y=acc_sig_history, name='Sigmoid', mode='lines+markers', line=dict(color='#ff0055', width=3)))
        fig1.add_trace(go.Scatter(x=list(range(1, epoch+1)), y=acc_relu_history, name='ReLU', mode='lines+markers', line=dict(color='#00ffcc', width=3)))
        fig1.update_layout(title=f"Epoch {epoch} | Sigmoid: {acc_sig:.2f} vs ReLU: {acc_relu:.2f}",
                           yaxis=dict(range=[0, 1.05], title="Accuracy (정확도)"), xaxis=dict(title="Epochs", range=[0, epochs+1]),
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"), height=350, margin=dict(l=20,r=20,t=40,b=20))
        plot_ph1.plotly_chart(fig1, use_container_width=True)
        
        # UI 업데이트 2: Dead ReLU 맵 (우측)
        if act_model:
            z_grid = np.array(z_map)
            fig2 = go.Figure(data=go.Heatmap(
                z=z_grid,
                colorscale=[[0, '#00ffcc'], [1, '#333344']],
                zmin=0, zmax=1,
                showscale=False,
                xgap=2, ygap=2
            ))
            fig2.update_layout(title=f"Dead ReLU 비율: {dead_ratio*100:.1f}% ({total_dead}/{total_neurons_cnt} 뉴런 사망)",
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(title="Hidden Layers (Input → Output)", showgrid=False, zeroline=False, showticklabels=False, autorange='reversed'),
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color="#e0e6ed"), height=350, margin=dict(l=20,r=20,t=40,b=20))
            plot_ph2.plotly_chart(fig2, use_container_width=True)
            
        progress.progress(epoch / epochs)
        time.sleep(0.01) # 애니메이션 프레임 지연
        
        # 동적 Nexus 대사 변경 (마지막 에포크 기준)
        if epoch == epochs:
            if dead_ratio > 0.3:
                nexus_ph.markdown('<div class="nexus-dialogue" style="border-left-color: #ff0055; color: #ff0055;">🤖 <b>Nexus:</b> 앗... 학습률(Learning Rate)이 너무 높아서 가중치가 과하게 업데이트되며 큰 음수 바이어스가 생겼어요. 수많은 뉴런들이 영원히 잠들어버리는 <b>Dead ReLU</b>가 대규모로 발생해 신호가 끊겼습니다!</div>', unsafe_allow_html=True)
            elif acc_relu > acc_sig + 0.1 and acc_relu >= 0.8:
                nexus_ph.markdown('<div class="nexus-dialogue" style="border-left-color: #00ffcc; color: #00ffcc;">🤖 <b>Nexus:</b> 엄청난 에너지가 느껴져요! 뉴런들이 죽지 않고 다시 깨어나서, 그 깊은 층을 통과하고도 완벽하게 패턴을 인식해냈어요! 이것이 ReLU의 힘이군요!</div>', unsafe_allow_html=True)
            elif acc_sig >= 0.8:
                nexus_ph.markdown('<div class="nexus-dialogue">🤖 <b>Nexus:</b> Sigmoid로도 학습이 되긴 하지만, 층이 더 깊어지면 위험할 수 있습니다. 계속 연구해봅시다!</div>', unsafe_allow_html=True)
            else:
                nexus_ph.markdown('<div class="nexus-dialogue">🤖 <b>Nexus:</b> 흐음... 아직 완벽하게 각성하진 못했어요. 층 수, 뉴런 수, 학습률 파라미터를 다시 조절해보세요!</div>', unsafe_allow_html=True)
                
    progress.empty()
    
    # --- 업적(별) 평가 로직 ---
    if acc_relu_history[-1] >= 0.85 and num_layers >= 5:
        stars = 1
        if dead_ratio < 0.1: stars += 1
        if neurons <= 32: stars += 1
        
        st.session_state.cleared = True
        st.session_state.stars = stars

# --- 클리어 메시지 출력 ---
if st.session_state.cleared:
    stars = st.session_state.stars
    st.balloons()
    st.success(f"### 🏆 쾌거! ReLU의 구원자 달성! (획득 별점: {'★'*stars}{'☆'*(3-stars)})")
    if stars == 3:
        st.info("완벽합니다! 모델의 파라미터(뉴런)를 낭비하지 않으면서도, 적절한 학습률을 세팅하여 Dead ReLU를 최소화하고 깊은 신경망의 학습을 성공적으로 이끌어냈습니다.")
    else:
        st.info("별 3개를 달성하려면, [은닉층 뉴런 수 32개 이하 (최적화)], [Dead ReLU 10% 미만 (안정적인 학습률)] 조건을 모두 만족한 채로 학습을 마쳐야 합니다!")
