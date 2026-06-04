import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import os

# TensorFlow 로그 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# --- UI 기본 설정 ---
st.set_page_config(page_title="Neural Odyssey: Stage 11", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
    }
    h1, h2, h3 {
        color: #ffaa00;
        text-shadow: 0 0 8px rgba(255, 170, 0, 0.5);
    }
    .nexus-dialogue {
        background: linear-gradient(90deg, rgba(255, 170, 0, 0.1) 0%, rgba(0, 255, 204, 0.05) 100%);
        border-left: 5px solid #ffaa00;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(255, 170, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 데이터 및 모델 셋업 (캐싱) ---
@st.cache_resource
def get_fashion_mnist_subset():
    # Fashion MNIST 로드 후, 빠른 연산을 위해 샘플링
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_train = X_train[:10000].reshape(-1, 28*28) / 255.0
    y_train = y_train[:10000]
    X_test = X_test[:2000].reshape(-1, 28*28) / 255.0
    y_test = y_test[:2000]
    return X_train, y_train, X_test, y_test

@st.cache_resource
def get_base_weights():
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model.get_weights()

def create_model_with_opt(opt):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.set_weights(get_base_weights()) # 모든 모델이 동일한 출발선에서 시작
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 사이드바 (Optimizer Tuning Center) ---
st.sidebar.title("Optimizer Tuning Center")
st.sidebar.markdown("다양한 옵티마이저를 레이스에 참가시키고 수렴 속도와 안정성을 비교하세요!")

selected_opts = st.sidebar.multiselect("경주 참가자 선택", ["SGD", "Momentum", "RMSprop", "Adam"], default=["SGD", "Adam"])

lr = st.sidebar.select_slider("글로벌 학습률 (Learning Rate)", options=[0.0001, 0.001, 0.01, 0.1], value=0.001, help="보폭의 크기입니다. 너무 크면 최적점을 지나치고(발산), 너무 작으면 도달하는 데 한세월이 걸립니다.")

momentum_val = 0.0
beta_1 = 0.9
beta_2 = 0.999

if "Momentum" in selected_opts:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Momentum 튜닝")
    momentum_val = st.sidebar.slider("관성 (Momentum)", 0.0, 0.99, 0.9, help="과거에 이동하던 방향을 얼마나 기억할지 결정합니다.")

if "Adam" in selected_opts:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adam 튜닝")
    beta_1 = st.sidebar.slider("Beta 1 (관성 계수)", 0.0, 0.99, 0.9, help="Momentum과 유사하게 이전 기울기를 기억합니다.")
    beta_2 = st.sidebar.slider("Beta 2 (적응형 계수)", 0.0, 0.999, 0.999, format="%.3f", help="RMSprop과 유사하게 기울기의 크기에 따라 학습률을 적응시킵니다.")

st.sidebar.markdown("---")
start_btn = st.sidebar.button("🏁 10 Epoch 레이스 시작!")

st.sidebar.markdown("---")
st.sidebar.subheader("🌟 미션 달성 조건 (별 3개)")
st.sidebar.markdown("- ★ : Adam 포함하여 레이스 완주\n- ★★ : Adam이 SGD보다 빠른 수렴을 입증 (최종 Loss 비교)\n- ★★★ : Adam 튜닝으로 Validation Accuracy **86%** 이상 달성")

# --- 메인 영역 (Game Area & Showdown Arena) ---
st.title("Neural Odyssey - Stage 11: 최적의 길 찾기 (Optimizer Showdown)")

nexus_ph = st.empty()
nexus_ph.markdown('<div class="nexus-dialogue">🤖 <b>Nexus:</b> 지형(Loss Landscape)이 너무 험난하고 복잡해요! 관성을 이용할까요, 아니면 보폭을 조절할까요? 저를 가장 빠르고 안정적으로 정답에 도달하도록 이끌어 줄 옵티마이저를 찾아주세요!</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏎️ Loss Curve Race")
    st.markdown("실제 신경망(Fashion MNIST)이 학습하면서 갱신하는 Validation Loss 궤적입니다.")
    chart_loss_ph = st.empty()
with col2:
    st.subheader("🗺️ Loss Landscape (이론적 궤적)")
    st.markdown("가상의 2D Loss 공간(계곡 형태)에서 각 알고리즘이 목적지(별)를 향해 나아가는 방식입니다.")
    chart_2d_ph = st.empty()

colors = {'SGD': '#aaaaaa', 'Momentum': '#ffcc00', 'RMSprop': '#ff0055', 'Adam': '#00ffcc'}

# --- 2D 시뮬레이션 환경 구성 ---
state_2d = {}
for name in selected_opts:
    state_2d[name] = {
        'pos': np.array([-8.0, 6.0]),
        'v': np.zeros(2),
        'm': np.zeros(2),
        't': 0,
        'history': [np.array([-8.0, 6.0])]
    }

def step_2d(name, user_lr, momentum_val, beta_1, beta_2):
    lr_2d = user_lr * 100 # 시각화를 위한 스케일링
    s = state_2d[name]
    x, y = s['pos']
    # f(x,y) = x^2 / 20 + y^2 에 대한 그레디언트
    grad = np.array([x / 10.0, 2.0 * y])
    
    if name == 'SGD':
        s['pos'] = s['pos'] - lr_2d * grad
    elif name == 'Momentum':
        s['v'] = momentum_val * s['v'] - lr_2d * grad
        s['pos'] = s['pos'] + s['v']
    elif name == 'RMSprop':
        s['v'] = 0.9 * s['v'] + 0.1 * (grad**2)
        s['pos'] = s['pos'] - (lr_2d / (np.sqrt(s['v']) + 1e-8)) * grad
    elif name == 'Adam':
        s['t'] += 1
        s['m'] = beta_1 * s['m'] + (1 - beta_1) * grad
        s['v'] = beta_2 * s['v'] + (1 - beta_2) * (grad**2)
        m_hat = s['m'] / (1 - beta_1**s['t'])
        v_hat = s['v'] / (1 - beta_2**s['t'])
        s['pos'] = s['pos'] - (lr_2d / (np.sqrt(v_hat) + 1e-8)) * m_hat
        
    s['pos'] = np.clip(s['pos'], -15, 15) # 화면 이탈 방지
    s['history'].append(s['pos'].copy())

def draw_2d_chart():
    x_grid = np.linspace(-10, 10, 100)
    y_grid = np.linspace(-10, 10, 100)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    Z_mesh = (X_mesh**2) / 20.0 + Y_mesh**2
    
    fig = go.Figure(data=go.Contour(
        z=Z_mesh, x=x_grid, y=y_grid,
        colorscale='Viridis',
        showscale=False,
        contours=dict(start=0, end=100, size=5, coloring='heatmap')
    ))
    
    for name in selected_opts:
        hist_2d = np.array(state_2d[name]['history'])
        fig.add_trace(go.Scatter(x=hist_2d[:, 0], y=hist_2d[:, 1], mode='lines+markers', name=name, line=dict(color=colors[name], width=2), marker=dict(size=4)))
        
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='red', symbol='star', size=15), name='Global Minimum'))
    
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20), xaxis=dict(range=[-10, 10]), yaxis=dict(range=[-10, 10]))
    return fig

# 초기 빈 화면에 2D 플롯 출력
chart_2d_ph.plotly_chart(draw_2d_chart(), use_container_width=True)

# --- 학습 로직 ---
if 'cleared' not in st.session_state:
    st.session_state.cleared = False
if 'stars' not in st.session_state:
    st.session_state.stars = 0

if start_btn:
    if len(selected_opts) == 0:
        st.warning("최소 1개의 옵티마이저를 선택해야 레이스를 시작할 수 있습니다!")
        st.stop()
        
    X_train, y_train, X_test, y_test = get_fashion_mnist_subset()
    
    tf.keras.backend.clear_session()
    
    models = {}
    opts_dict = {
        'SGD': tf.keras.optimizers.SGD(learning_rate=lr),
        'Momentum': tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum_val),
        'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=lr),
        'Adam': tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
    }
    
    for name in selected_opts:
        models[name] = create_model_with_opt(opts_dict[name])
        
    histories = {name: {'val_loss': [], 'val_acc': []} for name in selected_opts}
    epochs = 10
    
    progress = st.progress(0)
    
    for epoch in range(1, epochs + 1):
        for name in selected_opts:
            # 실 모델 학습 (1 에포크씩)
            h = models[name].fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=256, verbose=0)
            histories[name]['val_loss'].append(h.history['val_loss'][0])
            histories[name]['val_acc'].append(h.history['val_accuracy'][0])
            
            # 2D 시뮬레이션 (에포크당 8 스텝 전진)
            for _ in range(8): 
                step_2d(name, lr, momentum_val, beta_1, beta_2)
                
        # 좌측 차트: Keras 모델 Validation Loss 업데이트
        fig_loss = go.Figure()
        for name in selected_opts:
            fig_loss.add_trace(go.Scatter(y=histories[name]['val_loss'], name=name, mode='lines+markers', line=dict(color=colors[name], width=3)))
        fig_loss.update_layout(title="Validation Loss Race", template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20), xaxis=dict(title="Epoch", range=[0, epochs+1]), yaxis=dict(title="Loss", autorange='reversed')) # Loss는 낮을수록 좋으므로 뒤집기
        chart_loss_ph.plotly_chart(fig_loss, use_container_width=True)
        
        # 우측 차트: 2D 최적화 궤적 업데이트
        chart_2d_ph.plotly_chart(draw_2d_chart(), use_container_width=True)
        
        progress.progress(epoch / epochs)
        
        # 동적 텍스트 중계
        winner = min(histories.keys(), key=lambda k: histories[k]['val_loss'][-1])
        if epoch == epochs:
            nexus_ph.markdown(f'<div class="nexus-dialogue" style="border-left-color:{colors[winner]}; color:{colors[winner]};">🤖 <b>Nexus:</b> 🏁 레이스 종료! 가장 안정적이고 빠르게 최저 손실점에 도달한 옵티마이저는 **{winner}** 입니다! 관성과 적응력을 갖춘 기술이 승리했어요!</div>', unsafe_allow_html=True)
        else:
            nexus_ph.markdown(f'<div class="nexus-dialogue" style="border-left-color:{colors[winner]}; color:{colors[winner]};">🤖 <b>Nexus:</b> 레이스 진행 중... 현재 1등은 **{winner}**! 에포크 {epoch}/10 돌파!</div>', unsafe_allow_html=True)
            
        time.sleep(0.01)
        
    progress.empty()
    
    # --- 하단 대시보드 ---
    st.markdown("---")
    st.subheader("📊 성능 대시보드 (최종 결과)")
    cols = st.columns(len(selected_opts))
    for i, name in enumerate(selected_opts):
        final_acc = histories[name]['val_acc'][-1]
        final_loss = histories[name]['val_loss'][-1]
        cols[i].metric(f"{name} (정확도)", f"{final_acc*100:.2f}%", f"Loss: {final_loss:.4f}", delta_color="inverse")
        
    # --- 업적/별점 평가 로직 ---
    stars = 0
    if "Adam" in selected_opts:
        stars += 1
        if "SGD" in selected_opts:
            if histories['Adam']['val_loss'][-1] < histories['SGD']['val_loss'][-1]:
                stars += 1
        else:
            if histories['Adam']['val_acc'][-1] >= 0.80:
                stars += 1
                
        if histories['Adam']['val_acc'][-1] >= 0.86:
            stars += 1
            
    st.session_state.cleared = True
    st.session_state.stars = stars

# --- 클리어 메시지 출력 ---
if st.session_state.get('cleared', False):
    stars = st.session_state.stars
    st.balloons()
    st.success(f"### 🏆 쾌거! 최적화 마스터 달성! (별점: {'★'*stars}{'☆'*(3-stars)})")
    if stars == 3:
        st.info("완벽합니다! Adam의 강력함을 입증하고 하이퍼파라미터 조율을 통해 단 10 에포크라는 짧은 시간 안에 86% 이상의 놀라운 성능을 이끌어냈습니다. 현대 딥러닝에서 왜 Adam이 표준(Standard)으로 쓰이는지 직접 확인하셨습니다!")
    else:
        st.info("별 3개를 달성하려면 [Adam과 SGD를 함께 선택]하여 Adam의 우월함을 증명하고, [학습률을 조절하여 Adam의 Accuracy 86% 이상]을 달성해보세요! (힌트: LR=0.001이 Adam에게 최적인 경우가 많습니다)")
