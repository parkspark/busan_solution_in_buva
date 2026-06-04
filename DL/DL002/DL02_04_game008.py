import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --- 기본 페이지 설정 ---
st.set_page_config(page_title="Neural Odyssey: Stage 8", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS (Neon Dark Theme) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
    }
    h1, h2, h3 {
        color: #ff0055;
        text-shadow: 0 0 8px rgba(255, 0, 85, 0.5);
    }
    .nexus-dialogue {
        background: linear-gradient(90deg, rgba(255, 0, 85, 0.1) 0%, rgba(255, 255, 0, 0.05) 100%);
        border-left: 5px solid #ff0055;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(255, 0, 85, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 딥러닝 유틸리티 (NumPy Only) ---
def sigmoid(x):
    # 오버플로우 방지 클리핑
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

@st.cache_data
def get_data():
    np.random.seed(42)
    n_samples = 400
    t = np.linspace(0, 2 * np.pi, n_samples // 2)
    # 복잡한 동심원 구조 데이터 (이진 분류)
    x1 = 0.5 * np.cos(t) + np.random.randn(n_samples // 2) * 0.1
    y1 = 0.5 * np.sin(t) + np.random.randn(n_samples // 2) * 0.1
    x2 = 1.0 * np.cos(t) + np.random.randn(n_samples // 2) * 0.1
    y2 = 1.0 * np.sin(t) + np.random.randn(n_samples // 2) * 0.1
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).reshape(-1, 1)
    return X, y

def init_weights(layers):
    np.random.seed(42)
    W, b = [], []
    for i in range(len(layers) - 1):
        # Xavier/Glorot Initialization (Sigmoid에 적합)
        limit = np.sqrt(6 / (layers[i] + layers[i+1]))
        W.append(np.random.uniform(-limit, limit, (layers[i], layers[i+1])))
        b.append(np.zeros((1, layers[i+1])))
    return W, b

def draw_sigmoid_deriv():
    x = np.linspace(-10, 10, 100)
    y = sigmoid_deriv(x)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#00ffcc', width=3)))
    
    fig.add_shape(type="line", x0=-10, x1=10, y0=0.25, y1=0.25, line=dict(color="red", width=2, dash="dash"))
    fig.add_annotation(x=0, y=0.26, text="최댓값 = 0.25", showarrow=False, font=dict(color="red"))
    
    fig.update_layout(
        title="Sigmoid 도함수 (미분값)",
        xaxis_title="Input (z)",
        yaxis_title="Derivative",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#e0e6ed"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def draw_plots(layer_grads, epoch, loss, placeholder):
    fig = go.Figure()
    
    # x축 라벨: 레이어 이름 설정 (마지막은 Output)
    x_labels = [f"Layer {i+1}" if i < len(layer_grads)-1 else "Output" for i in range(len(layer_grads))]
    
    # 기울기 값에 따른 커스텀 색상 매핑 (초록 -> 노랑 -> 빨강 -> 검정)
    custom_colorscale = [
        [0.0, '#1a1a24'],  # 거의 검은색 (소멸, Dead)
        [0.1, '#ff0055'],  # 빨간색 (위험)
        [0.5, '#ffcc00'],  # 노란색 (경고)
        [1.0, '#00ffcc']   # 청록색 (건강)
    ]
    
    fig.add_trace(go.Bar(
        x=x_labels,
        y=layer_grads,
        marker=dict(
            color=layer_grads,
            colorscale=custom_colorscale,
            cmin=0.0,
            cmax=0.02, # 이 값을 기준으로 색상이 매핑됨
            line=dict(color='#ffffff', width=0.5)
        ),
        text=[f"{g:.5f}" if g >= 0.00001 else "0.0000" for g in layer_grads],
        textposition='outside',
        textfont=dict(color="white")
    ))
    
    fig.update_layout(
        title=f"Epoch: {epoch} | Loss: {loss:.4f}",
        yaxis=dict(title="평균 기울기 |Gradient|", range=[0, 0.025]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#e0e6ed"),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # st.empty() 플레이스홀더에 덮어쓰기
    placeholder.plotly_chart(fig, use_container_width=True)

# --- Session State ---
if 'cleared' not in st.session_state:
    st.session_state.cleared = False

# --- 사이드바 (Deep Network Architect) ---
st.sidebar.title("Deep Network Architect")
num_layers = st.sidebar.slider("은닉층(Hidden Layer) 개수", 1, 10, 5, help="층을 깊게 쌓을수록 기울기 소멸을 뚜렷하게 관찰할 수 있습니다.")
st.sidebar.text_input("활성화 함수 (Activation)", "Sigmoid (Locked 🔒)", disabled=True)
train_btn = st.sidebar.button("🧠 깊은 뇌 학습 시작")

# --- 메인 영역 (Game Area & Crisis Visualizer) ---
st.title("Neural Odyssey - Stage 8: 기울기 소멸의 위기 (Vanishing Gradient Crisis)")

nexus_placeholder = st.empty()
warning_placeholder = st.empty()

# 상단: Nexus의 호소 시스템
if num_layers <= 3:
    nexus_placeholder.markdown('<div class="nexus-dialogue" style="border-left-color: #00ffcc;">🤖 <b>Nexus:</b> 이 정도 깊이는 버틸 수 있어요! 앞쪽 층까지 오차가 잘 전달될 거예요. 학습을 시작해볼까요?</div>', unsafe_allow_html=True)
else:
    nexus_placeholder.markdown('<div class="nexus-dialogue">🤖 <b>Nexus:</b> 점점... 아무 생각도 안 나요... 앞쪽 뉴런들이 얼어붙은 것 같아요... 깊어질수록 오차(Gradient)가 희미해집니다...</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🔥 각 층의 기울기 (Gradient Flow)")
    plot_placeholder = st.empty()
    # 학습 시작 전 초기 빈 그래프
    draw_plots([0] * (num_layers + 1), 0, 1.0, plot_placeholder)

with col2:
    st.subheader("📉 Sigmoid 미분의 한계")
    st.plotly_chart(draw_sigmoid_deriv(), use_container_width=True)
    st.info("💡 **왜 앞쪽 층의 기울기가 사라질까?**\n\nSigmoid 함수의 도함수(미분값)는 아무리 입력값이 좋아도 **최대 0.25**입니다. 연쇄 법칙(Chain Rule)에 의해 역전파 시 각 층의 미분값이 계속 곱해집니다. 은닉층이 5개라면, 출력층의 오차가 맨 앞쪽 층으로 갈 때 기울기는 최대 $0.25^5 \\approx 0.00097$ 배로 축소되며 결국 0에 수렴하게 됩니다.")

# --- 학습 루프 (애니메이션) ---
if train_btn:
    # 모델 설계: 입력(2) -> 은닉층(16)*N -> 출력(1)
    layer_sizes = [2] + [16]*num_layers + [1]
    W, b = init_weights(layer_sizes)
    X, y = get_data()
    
    epochs = 150
    lr = 0.5
    m = X.shape[0]
    
    min_grad_found = False
    progress_bar = st.progress(0)
    
    for epoch in range(1, epochs + 1):
        # 1. 순전파 (Forward)
        activations = [X]
        for i in range(len(W)):
            z = np.dot(activations[-1], W[i]) + b[i]
            a = sigmoid(z)
            activations.append(a)
        
        a_out = activations[-1]
        loss = -np.mean(y * np.log(a_out + 1e-8) + (1 - y) * np.log(1 - a_out + 1e-8))
        
        # 2. 역전파 (Backward)
        dz = a_out - y # BCE + Sigmoid 미분
        dW = []
        db = []
        for i in reversed(range(len(W))):
            dW_i = np.dot(activations[i].T, dz) / m
            db_i = np.sum(dz, axis=0, keepdims=True) / m
            dW.append(dW_i)
            db.append(db_i)
            
            # 이전 층으로 오차 전파
            if i > 0:
                da = activations[i] * (1 - activations[i]) # Sigmoid 미분
                dz = np.dot(dz, W[i].T) * da # 연쇄 법칙
        
        dW.reverse()
        db.reverse()
        
        # 3. 가중치 업데이트
        for i in range(len(W)):
            W[i] -= lr * dW[i]
            b[i] -= lr * db[i]
            
        # 4. 각 층별 기울기 절댓값 평균 추출
        layer_grads = [np.mean(np.abs(dw)) for dw in dW]
        
        # UI 업데이트 (실시간 애니메이션)
        if epoch % 5 == 0 or epoch == 1:
            draw_plots(layer_grads, epoch, loss, plot_placeholder)
            progress_bar.progress(epoch / epochs)
            
            # 첫 번째 은닉층(Layer 1)의 기울기 감시
            first_layer_grad = layer_grads[0]
            if first_layer_grad <= 0.001 and num_layers >= 5:
                min_grad_found = True
                warning_placeholder.error("🚨 **[Vanishing Gradient Crisis!]** 맨 앞쪽(Layer 1)의 기울기가 0.001 이하로 소멸했습니다! 가중치가 더 이상 업데이트되지 않아 학습이 마비됩니다.")
            elif first_layer_grad > 0.001:
                warning_placeholder.empty()
                
            time.sleep(0.03) # 프레임 지연
            
    progress_bar.empty()
    
    # 5. 게임 클리어 조건 판단
    if num_layers >= 5 and min_grad_found:
        st.session_state.cleared = True

# --- 클리어 업적 표시 ---
if st.session_state.cleared:
    st.success("### 🏆 업적 달성: [위기의 목격자]")
    st.info("기울기가 소멸하여 학습이 마비되는 현상을 정확히 관측했습니다! 이로써 Sigmoid만을 사용하여 깊은 층을 쌓는 것의 치명적인 한계를 깨달았습니다. 다음 스테이지 **[Stage 9: ReLU의 구원]** 으로 가는 문이 열렸습니다.")
    st.balloons()
