import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- 기본 페이지 설정 ---
st.set_page_config(page_title="Neural Odyssey: Stage 7", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS (Neon Dark Theme) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
    }
    h1, h2, h3 {
        color: #00e5ff;
        text-shadow: 0 0 8px rgba(0, 229, 255, 0.5);
    }
    .nexus-dialogue {
        background: linear-gradient(90deg, rgba(0, 229, 255, 0.1) 0%, rgba(255, 0, 255, 0.05) 100%);
        border-left: 5px solid #ff00ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(255, 0, 255, 0.2);
    }
    div[data-testid="metric-container"] {
        background-color: rgba(0, 229, 255, 0.05);
        border: 1px solid rgba(0, 229, 255, 0.2);
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 딥러닝 함수 구현 (NumPy Only) ---
def sigmoid(x):
    # 오버플로우 방지를 위한 클리핑
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def forward(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def compute_loss(a2, y):
    # MSE Loss
    return np.mean((a2 - y)**2)

# --- 상태 초기화 (st.session_state) ---
if 'initialized' not in st.session_state:
    np.random.seed(42)
    # 가중치와 편향 초기화
    st.session_state.W1 = np.random.randn(2, 3) * 0.5
    st.session_state.b1 = np.zeros((1, 3))
    st.session_state.W2 = np.random.randn(3, 1) * 0.5
    st.session_state.b2 = np.zeros((1, 1))
    
    st.session_state.epoch = 0
    st.session_state.loss_history = []
    
    # 단순화된 데이터셋 (XOR 패턴 형태의 4사분면 분포)
    X = np.random.rand(100, 2) * 2 - 1
    y = (X[:, 0] * X[:, 1] > 0).astype(int).reshape(-1, 1)
    st.session_state.X = X
    st.session_state.y = y
    
    # 시각화용 변수
    st.session_state.current_a1 = np.zeros((3,))
    st.session_state.current_a2 = np.zeros((1,))
    st.session_state.current_dW1 = np.zeros((2, 3))
    st.session_state.current_dW2 = np.zeros((3, 1))
    st.session_state.chain_comps = None
    st.session_state.is_cleared = False
    st.session_state.initialized = True

# --- 학습 로직 (1 Step) ---
def train_step(lr, batch_size):
    # 미니배치 샘플링
    idx = np.random.choice(len(st.session_state.X), batch_size, replace=False)
    X_b = st.session_state.X[idx]
    y_b = st.session_state.y[idx]
    
    # 1. 순전파 (Forward Propagation)
    z1, a1, z2, a2 = forward(X_b, st.session_state.W1, st.session_state.b1, st.session_state.W2, st.session_state.b2)
    loss = compute_loss(a2, y_b)
    
    # 2. 역전파 (Backpropagation)
    m = batch_size
    # 출력층 오차 계산 (MSE의 미분: 2*(a2-y))
    dL_da2 = 2 * (a2 - y_b)
    da2_dz2 = a2 * (1 - a2)
    dL_dz2 = dL_da2 * da2_dz2
    
    # 출력층 가중치 기울기
    dW2 = np.dot(a1.T, dL_dz2) / m
    db2 = np.sum(dL_dz2, axis=0, keepdims=True) / m
    
    # 은닉층 오차 전파
    dL_da1 = np.dot(dL_dz2, st.session_state.W2.T)
    da1_dz1 = a1 * (1 - a1)
    dL_dz1 = dL_da1 * da1_dz1
    
    # 은닉층 가중치 기울기
    dW1 = np.dot(X_b.T, dL_dz1) / m
    db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m
    
    # 3. 가중치 업데이트
    st.session_state.W1 -= lr * dW1
    st.session_state.b1 -= lr * db1
    st.session_state.W2 -= lr * dW2
    st.session_state.b2 -= lr * db2
    
    st.session_state.epoch += 1
    st.session_state.loss_history.append(loss)
    
    # 시각화를 위해 평균 활성화 값 및 기울기 저장
    st.session_state.current_a1 = np.mean(a1, axis=0)
    st.session_state.current_a2 = np.mean(a2, axis=0)
    st.session_state.current_dW1 = dW1
    st.session_state.current_dW2 = dW2
    
    # Chain Rule 돋보기를 위해 배치의 '첫 번째' 데이터 분해값 저장
    st.session_state.chain_comps = {
        'dL_da2': dL_da2[0, 0],
        'da2_dz2': da2_dz2[0, 0],
        'a1': a1[0, :],
        'W2': st.session_state.W2[:, 0],
        'dL_da1': dL_da1[0, :],
        'da1_dz1': da1_dz1[0, :],
        'X': X_b[0, :]
    }

# --- 사이드바 (Training Control Center) ---
st.sidebar.title("Training Control Center")

lr = st.sidebar.slider("Learning Rate (lr)", 0.01, 2.0, 0.5, 0.01)
epochs = st.sidebar.number_input("Epochs (Auto Train용)", 1, 100, 10)
batch_size = st.sidebar.selectbox("Mini-batch size", [1, 16, 32, 100], index=2)

col1, col2 = st.sidebar.columns(2)
if col1.button("▶ 한 스텝 실행"):
    train_step(lr, batch_size)
if col2.button("⏩ 전체 학습"):
    for _ in range(epochs):
        train_step(lr, batch_size)

st.sidebar.markdown("---")
st.sidebar.subheader("가중치 해부 (Chain Rule)")
weight_choice = st.sidebar.selectbox("역전파를 분석할 가중치 선택:", [
    "W1 (Input 1 -> Hidden 1)", "W1 (Input 1 -> Hidden 2)", "W1 (Input 1 -> Hidden 3)",
    "W1 (Input 2 -> Hidden 1)", "W1 (Input 2 -> Hidden 2)", "W1 (Input 2 -> Hidden 3)",
    "W2 (Hidden 1 -> Output)", "W2 (Hidden 2 -> Output)", "W2 (Hidden 3 -> Output)"
])

if st.session_state.loss_history:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Loss History")
    st.sidebar.line_chart(st.session_state.loss_history)

if st.sidebar.button("🔄 초기화"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# --- 메인 영역 (Game Area & Nexus) ---
st.title("Neural Odyssey - Stage 7: 역전파의 비밀 (Backpropagation Mastery)")

current_loss = st.session_state.loss_history[-1] if st.session_state.loss_history else 1.0

# 클리어 조건 검사
if not st.session_state.is_cleared and current_loss <= 0.3 and st.session_state.epoch > 0:
    st.session_state.is_cleared = True
    st.balloons()

# 상단: Nexus 대사 및 미션
if st.session_state.is_cleared:
    dialogue = f"완벽해요! 오차가 0.3 이하로 떨어졌어요! (현재 Loss: {current_loss:.4f})"
    st.markdown(f'<div class="nexus-dialogue" style="border-left-color: #00ff00; color: #00ff00;">🤖 <b>Nexus:</b> {dialogue}</div>', unsafe_allow_html=True)
    
    stars = 1
    if st.session_state.epoch <= 20: stars += 1
    if batch_size >= 16: stars += 1
    
    st.success(f"### ⭐ 미션 클리어! 획득 별점: {'★' * stars}{'☆' * (3 - stars)}")
    st.write(f"- 기본 클리어 (Loss <= 0.3): ★")
    st.write(f"- 20 Epoch 이내 수렴 (현재 {st.session_state.epoch}): {'★' if st.session_state.epoch <= 20 else '☆'}")
    st.write(f"- 배치 사이즈 16 이상 사용 (현재 {batch_size}): {'★' if batch_size >= 16 else '☆'}")
else:
    dialogue = f"출력층에서 발생한 오차를 어떻게 줄일 수 있을 까요? <br><b>(미션: Loss를 0.3 이하로 낮추세요! 현재: {current_loss:.4f} | Epoch: {st.session_state.epoch})</b>"
    st.markdown(f'<div class="nexus-dialogue">🤖 <b>Nexus:</b> {dialogue}</div>', unsafe_allow_html=True)

# 중앙: 실시간 네트워크 그래프 시각화 (Plotly)
st.subheader("🌐 Network Flow & Gradients")
st.markdown("오른쪽(출력)에서 왼쪽(입력)으로 흐르는 **붉은색 엣지의 두께와 농도**가 기울기(Gradient)의 크기입니다. 노드는 활성화 값을 의미합니다.")

fig = go.Figure()

# 노드 좌표 설정 (2 -> 3 -> 1)
pos_x = [0, 0, 1, 1, 1, 2]
pos_y = [1, -1, 2, 0, -2, 0]
node_labels = ["Input 1", "Input 2", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]

# 노드 색상 (활성화 값 기준)
if st.session_state.chain_comps is not None:
    act_x = st.session_state.chain_comps['X']
    act_h = st.session_state.current_a1
    act_o = st.session_state.current_a2
    node_vals = [act_x[0], act_x[1], act_h[0], act_h[1], act_h[2], act_o[0]]
else:
    node_vals = [0, 0, 0, 0, 0, 0]

# W1 엣지 그리기
for i in range(2):
    for j in range(3):
        grad = st.session_state.current_dW1[i, j]
        weight = st.session_state.W1[i, j]
        
        # 기울기 크기에 따른 두께 및 색상 농도 설정
        grad_mag = min(abs(grad) * 50, 15)
        color_intensity = min(abs(grad) * 4, 1.0)
        
        color = f"rgba(255, 50, 50, {color_intensity})" if grad_mag > 0.1 else "rgba(100, 100, 100, 0.2)"
        width = 1 + grad_mag
        
        fig.add_trace(go.Scatter(
            x=[pos_x[i], pos_x[2+j]],
            y=[pos_y[i], pos_y[2+j]],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f"W: {weight:.3f}<br>Grad: {grad:.4f}",
            showlegend=False
        ))

# W2 엣지 그리기
for j in range(3):
    grad = st.session_state.current_dW2[j, 0]
    weight = st.session_state.W2[j, 0]
    
    grad_mag = min(abs(grad) * 50, 15)
    color_intensity = min(abs(grad) * 4, 1.0)
    
    color = f"rgba(255, 50, 50, {color_intensity})" if grad_mag > 0.1 else "rgba(100, 100, 100, 0.2)"
    width = 1 + grad_mag
    
    fig.add_trace(go.Scatter(
        x=[pos_x[2+j], pos_x[5]],
        y=[pos_y[2+j], pos_y[5]],
        mode='lines',
        line=dict(width=width, color=color),
        hoverinfo='text',
        text=f"W: {weight:.3f}<br>Grad: {grad:.4f}",
        showlegend=False
    ))

# 노드 그리기
fig.add_trace(go.Scatter(
    x=pos_x,
    y=pos_y,
    mode='markers+text',
    marker=dict(
        size=50,
        color=node_vals,
        colorscale='Electric',
        cmin=-1, cmax=1,
        showscale=False,
        line=dict(color='#00e5ff', width=2)
    ),
    text=node_labels,
    textposition="bottom center",
    textfont=dict(color="#e0e6ed", size=14),
    showlegend=False
))

fig.update_layout(
    height=400,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# --- 하단: Chain Rule 돋보기 ---
st.markdown("---")
st.subheader("🔍 Chain Rule 돋보기")

# 선택된 가중치 매핑
w_map = {
    "W1 (Input 1 -> Hidden 1)": ("W1", 0, 0),
    "W1 (Input 1 -> Hidden 2)": ("W1", 0, 1),
    "W1 (Input 1 -> Hidden 3)": ("W1", 0, 2),
    "W1 (Input 2 -> Hidden 1)": ("W1", 1, 0),
    "W1 (Input 2 -> Hidden 2)": ("W1", 1, 1),
    "W1 (Input 2 -> Hidden 3)": ("W1", 1, 2),
    "W2 (Hidden 1 -> Output)": ("W2", 0, 0),
    "W2 (Hidden 2 -> Output)": ("W2", 1, 0),
    "W2 (Hidden 3 -> Output)": ("W2", 2, 0),
}
w_type, i, j = w_map[weight_choice]

if st.session_state.chain_comps is not None:
    c = st.session_state.chain_comps
    st.write(f"**현재 선택된 가중치:** {weight_choice} (미니배치 내 첫 번째 데이터 기준 분해)")
    
    if w_type == "W2":
        term1 = c['dL_da2']
        term2 = c['da2_dz2']
        term3 = c['a1'][i]
        grad_val = term1 * term2 * term3
        
        st.latex(r"\frac{\partial L}{\partial w} = " +
                 r"\frac{\partial L}{\partial a_{out}} \cdot \frac{\partial a_{out}}{\partial z_{out}} \cdot \frac{\partial z_{out}}{\partial w}")
        
        cols = st.columns(4)
        cols[0].metric(r"∂L / ∂a_out (오차)", f"{term1:.4f}")
        cols[1].metric(r"∂a_out / ∂z_out (활성화 미분)", f"{term2:.4f}")
        cols[2].metric(r"∂z_out / ∂w (은닉층 출력값)", f"{term3:.4f}")
        cols[3].metric("Total Gradient (기울기)", f"{grad_val:.4f}", delta=f"{-lr * grad_val:.4f} (가중치 변화량)", delta_color="normal")
        
    else:
        term1 = c['dL_da1'][j]
        term2 = c['da1_dz1'][j]
        term3 = c['X'][i]
        grad_val = term1 * term2 * term3
        
        st.latex(r"\frac{\partial L}{\partial w} = " +
                 r"\frac{\partial L}{\partial a_{hidden}} \cdot " +
                 r"\frac{\partial a_{hidden}}{\partial z_{hidden}} \cdot " +
                 r"\frac{\partial z_{hidden}}{\partial w}")
        
        cols = st.columns(4)
        cols[0].metric(r"∂L / ∂a_hidden (전파된 오차)", f"{term1:.4f}")
        cols[1].metric(r"∂a_h / ∂z_h (활성화 미분)", f"{term2:.4f}")
        cols[2].metric(r"∂z_h / ∂w (입력값)", f"{term3:.4f}")
        cols[3].metric("Total Gradient (기울기)", f"{grad_val:.4f}", delta=f"{-lr * grad_val:.4f} (가중치 변화량)", delta_color="normal")
else:
    st.info("👈 왼쪽 사이드바에서 '한 스텝 실행'을 눌러 역전파를 발생시켜보세요!")
