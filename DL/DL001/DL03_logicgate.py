import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================


# Custom CSS for Neon styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #a371f7;
        text-shadow: 0 0 10px #a371f7;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Logic Gate Data Definitions
# ==========================================
GATE_DATA = {
    "AND Gate": {
        "X": np.array([[0,0], [0,1], [1,0], [1,1]]),
        "y": np.array([0, 0, 0, 1]),
        "desc": "| $x_1$ | $x_2$ | $y$ |\n|---|---|---|\n| 0 | 0 | 0 |\n| 0 | 1 | 0 |\n| 1 | 0 | 0 |\n| 1 | 1 | 1 |"
    },
    "OR Gate": {
        "X": np.array([[0,0], [0,1], [1,0], [1,1]]),
        "y": np.array([0, 1, 1, 1]),
        "desc": "| $x_1$ | $x_2$ | $y$ |\n|---|---|---|\n| 0 | 0 | 0 |\n| 0 | 1 | 1 |\n| 1 | 0 | 1 |\n| 1 | 1 | 1 |"
    },
    "XOR Gate": {
        "X": np.array([[0,0], [0,1], [1,0], [1,1]]),
        "y": np.array([0, 1, 1, 0]),
        "desc": "| $x_1$ | $x_2$ | $y$ |\n|---|---|---|\n| 0 | 0 | 0 |\n| 0 | 1 | 1 |\n| 1 | 0 | 1 |\n| 1 | 1 | 0 |"
    }
}

# ==========================================
# 3. Session State Initialization
# ==========================================
if 'stage2_init' not in st.session_state:
    st.session_state['w1'] = 0.0
    st.session_state['w2'] = 0.0
    st.session_state['bias'] = 0.0
    st.session_state['gate'] = "AND Gate"
    st.session_state['and_cleared'] = False
    st.session_state['or_cleared'] = False
    st.session_state['xor_failed'] = False
    st.session_state['message'] = "Nexus: 논리 게이트 챌린지에 오신 것을 환영합니다! AND와 OR를 먼저 완벽하게 분리해 보세요."
    st.session_state['msg_type'] = "info" # info, success, warning, error
    st.session_state['stage2_init'] = True

# ==========================================
# 4. Helper Functions
# ==========================================
def step_function(z):
    return np.where(z > 0, 1, 0)

# ==========================================
# 5. Sidebar: Control Panel
# ==========================================
with st.sidebar:
    st.title("🎛️ Logic Control")
    
    st.markdown("### 🧩 미니 챌린지")
    selected_gate = st.selectbox(
        "도전할 논리 게이트를 선택하세요", 
        ["AND Gate", "OR Gate", "XOR Gate"],
        index=["AND Gate", "OR Gate", "XOR Gate"].index(st.session_state['gate'])
    )
    
    # If gate changed manually by user
    if selected_gate != st.session_state['gate']:
        st.session_state['gate'] = selected_gate
        st.session_state['w1'] = 0.0
        st.session_state['w2'] = 0.0
        st.session_state['bias'] = 0.0
        if selected_gate == "XOR Gate":
            st.session_state['message'] = "Nexus: 이 데이터는 뭔가 이상해요... 어떻게 선을 그어야 하죠?"
            st.session_state['msg_type'] = "warning"
        else:
            st.session_state['message'] = f"Nexus: {selected_gate}를 학습할 준비가 되었습니다."
            st.session_state['msg_type'] = "info"
        st.rerun() # Ensure sliders reset immediately
            
    st.markdown("### 📖 진리표 (Truth Table)")
    st.markdown(GATE_DATA[selected_gate]["desc"])
    
    st.divider()
    
    st.markdown("### 🛠️ 수동 조작")
    st.session_state['w1'] = st.slider("Weight 1 ($w_1$)", -5.0, 5.0, value=float(st.session_state['w1']), step=0.1)
    st.session_state['w2'] = st.slider("Weight 2 ($w_2$)", -5.0, 5.0, value=float(st.session_state['w2']), step=0.1)
    st.session_state['bias'] = st.slider("Bias ($b$)", -5.0, 5.0, value=float(st.session_state['bias']), step=0.1)
    
    st.divider()
    
    st.markdown("### 🎓 학습 컨트롤")
    epochs = st.slider("Epochs (반복 횟수)", 1, 50, 20)
    learning_rate = 0.1
    train_clicked = st.button("Train Nexus 🚀", use_container_width=True)

# ==========================================
# 6. Main Area: Game & Visualization
# ==========================================
st.title("Neural Odyssey 🌌")
st.subheader("Stage 2: 논리의 한계 (The Limits of Logic)")

# Retrieve current state
X = GATE_DATA[selected_gate]["X"]
y = GATE_DATA[selected_gate]["y"]
w1 = st.session_state['w1']
w2 = st.session_state['w2']
bias = st.session_state['bias']

# Calculate accuracy based on current weights
z_current = w1 * X[:, 0] + w2 * X[:, 1] + bias
preds_current = step_function(z_current)
accuracy = np.mean(preds_current == y)

# Manual overrides check
if not train_clicked:
    if accuracy == 1.0:
        if selected_gate == "AND Gate":
            st.session_state['and_cleared'] = True
        elif selected_gate == "OR Gate":
            st.session_state['or_cleared'] = True
            
        if selected_gate in ["AND Gate", "OR Gate"]:
            st.session_state['message'] = f"Nexus: 직선 하나로 완벽하게 나눴어요! {selected_gate}의 논리가 이해됩니다."
            st.session_state['msg_type'] = "success"
    else:
        if selected_gate == "XOR Gate":
            st.session_state['message'] = "Nexus: 이 데이터는 뭔가 이상해요... 한 번의 직선으로 나눌 수 있을까요?"
            st.session_state['msg_type'] = "warning"
        else:
            st.session_state['message'] = "Nexus: 가중치를 조작하여 점들을 진리표에 맞게 완벽하게 나누는 선을 찾아보세요!"
            st.session_state['msg_type'] = "info"

# Nexus Dialogue Box
msg = st.session_state['message']
msg_type = st.session_state['msg_type']
if msg_type == "success":
    st.success(f"**Nexus**: {msg}")
elif msg_type == "warning":
    st.warning(f"**Nexus**: {msg}")
elif msg_type == "error":
    st.error(f"**Nexus**: {msg}")
else:
    st.info(f"**Nexus**: {msg}")

# Progression tracker
col1, col2, col3 = st.columns(3)
col1.checkbox("AND Gate 마스터", value=st.session_state['and_cleared'], disabled=True)
col2.checkbox("OR Gate 마스터", value=st.session_state['or_cleared'], disabled=True)
col3.checkbox("XOR Gate의 진실", value=st.session_state['xor_failed'], disabled=True)

# Placeholder for plotting
status_text = st.empty()
plot_placeholder = st.empty()

def draw_logic_boundary(w1_val, w2_val, b_val, current_gate):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    x_range = np.arange(x_min, x_max, 0.02)
    y_range = np.arange(y_min, y_max, 0.02)
    xx, yy = np.meshgrid(x_range, y_range)
                         
    Z = w1_val * xx + w2_val * yy + b_val
    Z = step_function(Z)
        
    fig = go.Figure()
    
    # Decision Boundary (Contour)
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale='RdBu',
        opacity=0.3,
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Grid lines to show (0,0), (0,1) etc clearly
    fig.add_shape(type="line", x0=0, y0=-0.5, x1=0, y1=1.5, line=dict(color="gray", width=1, dash="dot"))
    fig.add_shape(type="line", x0=1, y0=-0.5, x1=1, y1=1.5, line=dict(color="gray", width=1, dash="dot"))
    fig.add_shape(type="line", x0=-0.5, y0=0, x1=1.5, y1=0, line=dict(color="gray", width=1, dash="dot"))
    fig.add_shape(type="line", x0=-0.5, y0=1, x1=1.5, y1=1, line=dict(color="gray", width=1, dash="dot"))
    
    # Scatter Data - Blue Class (0)
    x_blue = X[y==0, 0]
    y_blue = X[y==0, 1]
    fig.add_trace(go.Scatter(
        x=x_blue, y=y_blue,
        mode='markers+text',
        marker=dict(color='#3182bd', size=25, line=dict(color='white', width=2)),
        name='False (0)',
        text=[f"({int(px)},{int(py)})" for px, py in zip(x_blue, y_blue)],
        textposition="top center",
        textfont=dict(color="white", size=16)
    ))
    
    # Scatter Data - Red Class (1)
    x_red = X[y==1, 0]
    y_red = X[y==1, 1]
    fig.add_trace(go.Scatter(
        x=x_red, y=y_red,
        mode='markers+text',
        marker=dict(color='#e6550d', size=25, line=dict(color='white', width=2)),
        name='True (1)',
        text=[f"({int(px)},{int(py)})" for px, py in zip(x_red, y_red)],
        textposition="top center",
        textfont=dict(color="white", size=16)
    ))
    
    fig.update_layout(
        title=f"2D 시각화: {current_gate} 결정 경계",
        xaxis_title="Input 1 ($x_1$)",
        yaxis_title="Input 2 ($x_2$)",
        xaxis=dict(range=[-0.5, 1.5], tickvals=[0, 1]),
        yaxis=dict(range=[-0.5, 1.5], tickvals=[0, 1]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500,
        showlegend=True
    )
    return fig

# Initial Render
plot_placeholder.plotly_chart(draw_logic_boundary(w1, w2, bias, selected_gate), use_container_width=True, key="plot_initial")

# ==========================================
# 7. Training Logic (Animation & Intentional Failure)
# ==========================================
if train_clicked:
    st.session_state['message'] = "Nexus가 학습 중입니다..."
    st.session_state['msg_type'] = "info"
    
    w1_t = w1
    w2_t = w2
    b_t = bias
    
    with st.spinner("Nexus가 진리표를 암기하고 있습니다..."):
        for epoch in range(epochs):
            status_text.markdown(f"**Training Progress:** Epoch {epoch+1}/{epochs}")
            errors = 0
            
            # For XOR, shuffle to make oscillation more visible
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            
            for i in indices:
                xi = X[i]
                yi = y[i]
                
                z_i = w1_t * xi[0] + w2_t * xi[1] + b_t
                y_hat = 1 if z_i > 0 else 0 
                
                update = learning_rate * (yi - y_hat)
                if update != 0.0:
                    w1_t += update * xi[0]
                    w2_t += update * xi[1]
                    b_t += update
                    errors += 1
                    
            plot_placeholder.plotly_chart(draw_logic_boundary(w1_t, w2_t, b_t, selected_gate), use_container_width=True, key=f"plot_train_{epoch}")
            time.sleep(0.15)
            
            if errors == 0:
                status_text.success(f"Training Early Stopped: {selected_gate} 완벽 분리!")
                break
                
    st.session_state['w1'] = float(w1_t)
    st.session_state['w2'] = float(w2_t)
    st.session_state['bias'] = float(b_t)
    
    # Final check
    z_final = w1_t * X[:, 0] + w2_t * X[:, 1] + b_t
    final_acc = np.mean(step_function(z_final) == y)
    
    if final_acc == 1.0:
        st.session_state['msg_type'] = "success"
        st.session_state['message'] = f"Nexus: 직선 하나로 완벽하게 나눴어요! {selected_gate}의 논리가 이해됩니다."
        if selected_gate == "AND Gate":
            st.session_state['and_cleared'] = True
        elif selected_gate == "OR Gate":
            st.session_state['or_cleared'] = True
    else:
        if selected_gate == "XOR Gate":
            st.session_state['msg_type'] = "error"
            st.session_state['message'] = "Nexus: 안 돼요! 아무리 선을 돌려도 한 번에 나눌 수 없어요... 제 뇌(단일 층)가 너무 얕은 걸까요? (선형 분리 불가)"
            st.session_state['xor_failed'] = True
        else:
            st.session_state['msg_type'] = "warning"
            st.session_state['message'] = "Nexus: 학습을 완료했지만 아직 완벽하지 않아요. 에포크를 늘려서 다시 훈련시켜 주세요."
            
    st.rerun()

# ==========================================
# 8. Unlocking Next Stage
# ==========================================
if st.session_state['and_cleared'] and st.session_state['or_cleared'] and st.session_state['xor_failed']:
    st.markdown("---")
    st.markdown("### 🔓 숨겨진 진실 도달 (Stage Clear)")
    st.success("**단층 신경망의 수학적 한계(Minsky & Papert, 1969)**를 직접 증명하셨습니다! 오직 다층 신경망(MLP)만이 XOR 문제를 해결할 수 있습니다.")
    if st.button("🚀 Stage 3: 다층 퍼셉트론(MLP)으로 나아가기"):
        st.balloons()
        st.switch_page("DL001/DL04_game.py")
