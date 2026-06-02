import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_blobs
import time

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================


# Custom CSS for Neon/Dark styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff;
        text-shadow: 0 0 10px #58a6ff;
    }
    .stProgress .st-bo {
        background-color: #238636;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State Initialization
# ==========================================
if 'init_done' not in st.session_state:
    # 선형 분리 가능한 2D 데이터 생성
    X, y = make_blobs(
        n_samples=100, 
        n_features=2, 
        centers=[[-2, -2], [2, 2]], 
        random_state=42, 
        cluster_std=0.8
    )
    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['w1'] = 0.0
    st.session_state['w2'] = 0.0
    st.session_state['bias'] = 0.0
    st.session_state['xp'] = 0
    st.session_state['cleared'] = False
    st.session_state['message'] = "Nexus: 안녕하세요, 수호자님! 저에게 첫 번째 지식을 불어넣어 주세요. 슬라이더를 움직이거나 학습을 시작해 보세요."
    st.session_state['init_done'] = True

# ==========================================
# 3. Helper Functions
# ==========================================
def step_function(z):
    return np.where(z >= 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ==========================================
# 4. Sidebar: Control Panel
# ==========================================
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.markdown("""
    ### 📖 개념 사전
    **퍼셉트론(Perceptron)**은 인공 신경망의 가장 기본 단위입니다.
    수식: $y = f(w_1 x_1 + w_2 x_2 + b)$
    * $x_1, x_2$: 입력 데이터 (특성)
    * $w_1, w_2$: 가중치 (Weights)
    * $b$: 편향 (Bias)
    * $f$: 활성화 함수 (Activation Function)
    """)
    
    st.divider()
    
    activation_choice = st.selectbox("활성화 함수 선택", ["Step Function", "Sigmoid"])
    
    st.markdown("### 🛠️ 수동 조작")
    st.slider("Weight 1 ($w_1$)", -5.0, 5.0, step=0.1, key='w1')
    st.slider("Weight 2 ($w_2$)", -5.0, 5.0, step=0.1, key='w2')
    st.slider("Bias ($b$)", -5.0, 5.0, step=0.1, key='bias')
    
    st.divider()
    
    st.markdown("### 🎓 학습 컨트롤")
    epochs = st.slider("Epochs (반복 횟수)", 1, 50, 10)
    learning_rate = 0.1
    train_clicked = st.button("Train Nexus 🚀", use_container_width=True)

# ==========================================
# 5. Main Area: Game & Visualization
# ==========================================
st.title("Neural Odyssey 🌌")
st.subheader("Stage 1: 퍼셉트론의 탄생 (Birth of the Perceptron)")

# Retrieve current state
X = st.session_state['X']
y = st.session_state['y']
w1 = st.session_state['w1']
w2 = st.session_state['w2']
bias = st.session_state['bias']

# Calculate accuracy based on current weights
z_current = w1 * X[:, 0] + w2 * X[:, 1] + bias
if activation_choice == "Step Function":
    preds_current = step_function(z_current)
else:
    preds_current = np.where(sigmoid(z_current) >= 0.5, 1, 0)
    
accuracy = np.mean(preds_current == y)

# Detect state changes dynamically (Manual overrides)
if not train_clicked:
    if accuracy == 1.0 and not st.session_state['cleared']:
        st.session_state['cleared'] = True
        st.session_state['xp'] += 100
        st.session_state['message'] = "Nexus: 세상에, 수호자님이 직접 완벽한 경계를 찾으셨군요! 제 안의 뉴런이 연결되었습니다. (XP +100)"
    elif accuracy < 1.0 and st.session_state['cleared']:
        st.session_state['cleared'] = False
        st.session_state['message'] = "Nexus: 경계가 다시 흐트러졌어요! 다시 조정해 주세요."
    elif accuracy < 1.0:
        st.session_state['message'] = "Nexus: 가중치를 조작하여 빨간 점과 파란 점을 완벽하게 나누는 선을 찾아보세요!"

# Nexus Dialogue Box
if st.session_state['cleared']:
    st.success(f"**Nexus**: {st.session_state['message']}")
else:
    st.info(f"**Nexus**: {st.session_state['message']}")

# Status Bar
col1, col2 = st.columns([3, 1])
with col1:
    st.progress(accuracy, text=f"Nexus Accuracy: {accuracy*100:.1f}%")
with col2:
    st.markdown(f"<h3 style='text-align: right; margin-top: 0;'>⭐ XP: {st.session_state['xp']}</h3>", unsafe_allow_html=True)

# Placeholder for plotting & status
status_text = st.empty()
plot_placeholder = st.empty()

def draw_boundary(w1_val, w2_val, b_val):
    """Plotly를 이용해 결정 경계와 데이터를 그리는 함수"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create Grid
    x_range = np.arange(x_min, x_max, 0.05)
    y_range = np.arange(y_min, y_max, 0.05)
    xx, yy = np.meshgrid(x_range, y_range)
                         
    Z = w1_val * xx + w2_val * yy + b_val
    if activation_choice == "Step Function":
        Z = step_function(Z)
    else:
        Z = sigmoid(Z)
        
    fig = go.Figure()
    
    # Decision Boundary (Contour)
    fig.add_trace(go.Contour(
        x=x_range,
        y=y_range,
        z=Z,
        colorscale='RdBu',
        opacity=0.4,
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Scatter Data - Blue Class (0)
    fig.add_trace(go.Scatter(
        x=X[y==0, 0], y=X[y==0, 1],
        mode='markers',
        marker=dict(color='#3182bd', size=12, line=dict(color='white', width=1.5)),
        name='Blue Class (0)'
    ))
    # Scatter Data - Red Class (1)
    fig.add_trace(go.Scatter(
        x=X[y==1, 0], y=X[y==1, 1],
        mode='markers',
        marker=dict(color='#e6550d', size=12, line=dict(color='white', width=1.5)),
        name='Red Class (1)'
    ))
    
    fig.update_layout(
        title="2D 시각화: 뉴런의 판단 경계",
        xaxis_title="Feature 1 ($x_1$)",
        yaxis_title="Feature 2 ($x_2$)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c9d1d9'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500
    )
    return fig

# Initial Render
plot_placeholder.plotly_chart(draw_boundary(w1, w2, bias), use_container_width=True)

# ==========================================
# 6. Training Logic (Animation)
# ==========================================
if train_clicked:
    st.session_state['cleared'] = False
    
    # Start training from current weights
    w1_t = w1
    w2_t = w2
    b_t = bias
    
    perfect = False
    
    with st.spinner("Nexus가 데이터를 학습하고 있습니다..."):
        for epoch in range(epochs):
            status_text.markdown(f"**Training Progress:** Epoch {epoch+1}/{epochs}")
            errors = 0
            
            for i in range(len(X)):
                xi = X[i]
                yi = y[i]
                
                # Forward pass
                z_i = w1_t * xi[0] + w2_t * xi[1] + b_t
                y_hat = 1 if z_i >= 0 else 0 # Perceptron learning rule
                
                # Weight update
                update = learning_rate * (yi - y_hat)
                if update != 0.0:
                    w1_t += update * xi[0]
                    w2_t += update * xi[1]
                    b_t += update
                    errors += 1
                    
            # Draw real-time update per epoch
            plot_placeholder.plotly_chart(draw_boundary(w1_t, w2_t, b_t), use_container_width=True)
            time.sleep(0.1) # Animation speed (0.1초 대기)
            
            if errors == 0:
                perfect = True
                status_text.success("Training Early Stopped: Perfect Separation! 🎉")
                break
                
    # Save results to session state
    st.session_state['w1'] = float(w1_t)
    st.session_state['w2'] = float(w2_t)
    st.session_state['bias'] = float(b_t)
    
    # Final check
    z_final = w1_t * X[:, 0] + w2_t * X[:, 1] + b_t
    final_acc = np.mean(step_function(z_final) == y)
    
    if final_acc == 1.0:
        st.session_state['cleared'] = True
        st.session_state['xp'] += 100
        st.session_state['message'] = "Nexus: 완벽해요! 학습 알고리즘을 통해 데이터를 깨끗하게 분류하는 법을 배웠습니다! (XP +100)"
    else:
        st.session_state['cleared'] = False
        st.session_state['message'] = "Nexus: 학습을 진행했지만 아직 완벽하지 않아요. 에포크를 늘리거나 초기값을 바꿔서 다시 훈련시켜 주세요."
        
    st.rerun() # Update UI fully

# ==========================================
# 7. Interactive Coding Area
# ==========================================
st.markdown("---")
st.markdown("### 💻 인터랙티브 코딩 (의사코드 테스트)")
st.caption("아래 공간에 퍼셉트론의 동작 방식을 파이썬 코드로 상상해서 적어보세요.")
code_input = st.text_area(
    "Code Editor", 
"""def predict_neuron(x1, x2, w1, w2, bias):
    # 1. 입력값과 가중치를 곱하고 편향을 더합니다.
    z = (w1 * x1) + (w2 * x2) + bias
    
    # 2. 활성화 함수(Step Function)를 적용합니다.
    if z >= 0:
        return 1  # Red Class
    else:
        return 0  # Blue Class
""", height=200)
