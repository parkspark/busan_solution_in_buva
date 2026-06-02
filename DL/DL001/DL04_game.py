import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================


st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    h1, h2, h3 { color: #f28b82; text-shadow: 0 0 10px #f28b82; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Simple Neural Network (NumPy)
# ==========================================
class SimpleMLP:
    """간단한 2층 신경망 (Input -> Hidden -> Output)"""
    def __init__(self, n_inputs, n_hidden, activation="Sigmoid", lr=0.1):
        # Xavier Initialization
        self.W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2. / n_inputs)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(2. / n_hidden)
        self.b2 = np.zeros((1, 1))
        
        self.activation = activation
        self.lr = lr
        
    def act_fn(self, z):
        if self.activation == "Sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.activation == "ReLU":
            return np.maximum(0, z)
        else: # None (Linear)
            return z
            
    def act_derivative(self, z, a):
        if self.activation == "Sigmoid":
            return a * (1 - a)
        elif self.activation == "ReLU":
            return (z > 0).astype(float)
        else: # None (Linear)
            return np.ones_like(z)
            
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
        
    def forward(self, X):
        # Hidden Layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.act_fn(self.z1)
        
        # Output Layer (Binary Classification -> Sigmoid)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
        
    def backward(self, X, y, output):
        m = X.shape[0]
        # BCE Loss derivative w.r.t z2
        dz2 = output - y.reshape(-1, 1)
        
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.act_derivative(self.z1, self.a1)
        
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Gradient Descent Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # Binary Cross Entropy
        return -np.mean(y_true.reshape(-1,1) * np.log(y_pred) + (1 - y_true.reshape(-1,1)) * np.log(1 - y_pred))

# ==========================================
# 3. UI Setup & Sidebar
# ==========================================
with st.sidebar:
    st.title("🛠️ Layer Builder")
    st.markdown("은닉층(Hidden Layer)을 설계하세요.")
    
    n_hidden = st.slider("은닉층 뉴런 수", 2, 8, 4)
    activation = st.selectbox("활성화 함수 (Activation)", ["ReLU", "Sigmoid", "None (Linear)"])
    
    st.divider()
    
    lr = st.slider("학습률 (Learning Rate)", 0.01, 1.0, 0.5, step=0.01)
    epochs = st.slider("Epochs", 100, 3000, 1000, step=100)
    
    train_clicked = st.button("Train Neural Network 🚀", use_container_width=True)

# ==========================================
# 4. Session State Management
# ==========================================
if 'stage3_init' not in st.session_state:
    st.session_state['trained_mlp'] = None
    st.session_state['loss_history'] = []
    st.session_state['status_msg'] = "**Nexus**: 사이드바에서 은닉층의 구조를 설계하고 학습을 시작해주세요!"
    st.session_state['status_type'] = "info"
    st.session_state['current_params'] = (n_hidden, activation, lr, epochs)
    st.session_state['stage3_init'] = True

# Detect parameter changes to reset view
if not train_clicked and st.session_state['current_params'] != (n_hidden, activation, lr, epochs):
    st.session_state['current_params'] = (n_hidden, activation, lr, epochs)
    st.session_state['trained_mlp'] = None
    st.session_state['loss_history'] = []
    st.session_state['status_msg'] = "**Nexus**: 설정이 변경되었습니다. 어떤 곡선이 그려질지 'Train' 버튼을 눌러 확인해보세요!"
    st.session_state['status_type'] = "info"
    st.rerun()

# ==========================================
# 5. Main Content Layout
# ==========================================
st.title("Neural Odyssey 🌌")
st.subheader("Stage 3: 숨겨진 층의 힘 (The Power of Hidden Layers)")

# XOR Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

msg_placeholder = st.empty()

col1, col2 = st.columns(2)
plot_left = col1.empty()
plot_right = col2.empty()

def draw_plots(mlp_model, loss_hist):
    # Contour Plot (Decision Boundary)
    x_range = np.linspace(-0.5, 1.5, 60)
    y_range = np.linspace(-0.5, 1.5, 60)
    xx, yy = np.meshgrid(x_range, y_range)
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    if mlp_model is None:
        np.random.seed(42) # fixed seed for reproducible dummy
        dummy = SimpleMLP(2, n_hidden, activation, lr)
        Z = dummy.forward(grid).reshape(xx.shape)
        np.random.seed(None)
    else:
        Z = mlp_model.forward(grid).reshape(xx.shape)
        
    fig1 = go.Figure()
    fig1.add_trace(go.Contour(x=x_range, y=y_range, z=Z, colorscale='RdBu', opacity=0.5, showscale=False, zmin=0, zmax=1, hoverinfo='skip'))
    
    # Data points
    fig1.add_trace(go.Scatter(x=X[y==0, 0], y=X[y==0, 1], mode='markers+text',
                              marker=dict(color='#3182bd', size=20, line=dict(color='white', width=2)), name='Class 0', text=['(0,0)','(1,1)'], textposition='top center', textfont=dict(color="white", size=14)))
    fig1.add_trace(go.Scatter(x=X[y==1, 0], y=X[y==1, 1], mode='markers+text',
                              marker=dict(color='#e6550d', size=20, line=dict(color='white', width=2)), name='Class 1', text=['(0,1)','(1,0)'], textposition='top center', textfont=dict(color="white", size=14)))
                              
    fig1.update_layout(title="Decision Boundary (결정 경계)", xaxis_title="Input 1 ($x_1$)", yaxis_title="Input 2 ($x_2$)", height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'), margin=dict(l=20, r=20, t=40, b=20), xaxis=dict(range=[-0.5, 1.5], tickvals=[0,1]), yaxis=dict(range=[-0.5, 1.5], tickvals=[0,1]))
    
    # Loss Curve Plot
    fig2 = go.Figure()
    if len(loss_hist) > 0:
        fig2.add_trace(go.Scatter(y=loss_hist, mode='lines', line=dict(color='#f28b82', width=3), name='BCE Loss'))
    else:
        fig2.add_trace(go.Scatter(y=[0], mode='lines', line=dict(color='rgba(0,0,0,0)'))) # Empty placeholder
        
    fig2.update_layout(title="Loss Curve (오차율 감소)", xaxis_title="Epochs", yaxis_title="Loss (BCE)", height=450, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'), margin=dict(l=20, r=20, t=40, b=20))
    if len(loss_hist) > 0:
        fig2.update_xaxes(range=[0, len(loss_hist)])
        
    return fig1, fig2

# ==========================================
# 6. Training Logic & Animation
# ==========================================
if train_clicked:
    mlp = SimpleMLP(n_inputs=2, n_hidden=n_hidden, activation=activation, lr=lr)
    loss_hist = []
    
    msg_placeholder.info("**Nexus**: 다층 신경망을 통한 학습을 진행하고 있습니다. 경계선이 어떻게 휘어지는지 지켜보세요!")
    
    # 30 frames for smooth animation
    update_freq = max(1, epochs // 30) 
    
    for epoch in range(1, epochs + 1):
        preds = mlp.forward(X)
        loss = mlp.compute_loss(y, preds)
        mlp.backward(X, y, preds)
        loss_hist.append(loss)
        
        # Real-time UI update
        if epoch % update_freq == 0 or epoch == epochs:
            f1, f2 = draw_plots(mlp, loss_hist)
            plot_left.plotly_chart(f1, use_container_width=True)
            plot_right.plotly_chart(f2, use_container_width=True)
            time.sleep(0.02)
            
    # Final Eval
    final_preds = (mlp.forward(X) >= 0.5).flatten()
    acc = np.mean(final_preds == y)
    
    st.session_state['trained_mlp'] = mlp
    st.session_state['loss_history'] = loss_hist
    
    if activation == "None (Linear)":
        st.session_state['status_type'] = "warning"
        st.session_state['status_msg'] = "**Nexus**: 층은 늘어났는데 제 생각(결정 경계)은 여전히 딱딱한 직선이에요! 비선형 활성화 함수가 필요한 것 같아요. 선형 변환은 아무리 곱해도 결국 선형일 뿐입니다."
    elif acc == 1.0:
        st.session_state['status_type'] = "success"
        st.session_state['status_msg'] = "**Nexus**: 놀라워요! 제 뇌가 곡선으로 공간을 분리하기 시작했어요! XOR의 비밀을 풀었습니다! 🎉"
    else:
        st.session_state['status_type'] = "error"
        st.session_state['status_msg'] = f"**Nexus**: 아직 곡선이 완벽하게 공간을 분리하지 못했어요. (정확도: {acc*100:.0f}%) 뉴런 수나 에포크, 학습률을 조절해보세요!"
        
    st.rerun()

# ==========================================
# 7. Render Outside Training Loop
# ==========================================
if not train_clicked:
    msg = st.session_state['status_msg']
    mtype = st.session_state['status_type']
    
    if mtype == "success": msg_placeholder.success(msg)
    elif mtype == "warning": msg_placeholder.warning(msg)
    elif mtype == "error": msg_placeholder.error(msg)
    else: msg_placeholder.info(msg)
    
    f1, f2 = draw_plots(st.session_state['trained_mlp'], st.session_state['loss_history'])
    plot_left.plotly_chart(f1, use_container_width=True)
    plot_right.plotly_chart(f2, use_container_width=True)
    
    if mtype == "success":
        st.markdown("---")
        st.markdown("### 🏆 Stage 3 Clear!")
        st.success("**비선형성의 마법(The Magic of Non-linearity)**을 완벽하게 이해하셨습니다! 은닉층과 비선형 활성화 함수가 결합되어 비로소 인공지능이 복잡한 패턴을 학습할 수 있게 되었습니다.")
        if st.button("🔑 Stage 4 으로 나아가기", use_container_width=True):
            st.balloons()
            st.switch_page("DL001/DL05_game.py")
