import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import time
import os

# Suppress TensorFlow logging to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================


st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    h1, h2, h3 { color: #58a6ff; text-shadow: 0 0 10px #58a6ff; }
    .nexus-dialogue { 
        border-left: 5px solid #a371f7; 
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 20px;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading & Caching
# ==========================================
@st.cache_data
def get_data():
    digits = load_digits()
    # 0~16 범위를 0~1로 스케일링하여 최적화 속도 향상
    X = digits.data / 16.0 
    y = digits.target
    images = digits.images
    return train_test_split(X, y, images, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test, img_train, img_test = get_data()

# ==========================================
# 3. Sidebar (Control Panel)
# ==========================================
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.subheader("데이터 미리보기 (8x8 Digits)")
    # 무작위 샘플 4개 추출
    np.random.seed(42) # 고정 시드로 UI 깜빡임 방지
    idx = np.random.choice(len(img_train), 4, replace=False)
    np.random.seed(None)
    
    fig_imgs = make_subplots(rows=1, cols=4, subplot_titles=[f"[{y_train[i]}]" for i in idx])
    for c, i in enumerate(idx):
        # 상하반전 복구 (np.flipud)
        fig_imgs.add_trace(go.Heatmap(z=np.flipud(img_train[i]), colorscale='gray', showscale=False), row=1, col=c+1)
        
    fig_imgs.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig_imgs.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    st.plotly_chart(fig_imgs, use_container_width=True)
    
    st.divider()
    
    st.subheader("학습 파라미터")
    st.markdown("두 모델 모두에게 동일한 반복 횟수(기회)를 제공합니다.")
    epochs = st.slider("Max Iterations / Epochs", 10, 500, 200, step=10)
    train_clicked = st.button("두 세계의 모델 학습하기 (Train Both) 🚀", use_container_width=True)

# ==========================================
# 4. State Management
# ==========================================
if 'stage5_init' not in st.session_state:
    st.session_state['stage5_init'] = True
    st.session_state['trained'] = False
    st.session_state['current_epochs'] = epochs

# 슬라이더 변경 시 결과 리셋
if st.session_state['current_epochs'] != epochs and not train_clicked:
    st.session_state['current_epochs'] = epochs
    st.session_state['trained'] = False
    st.rerun()

# ==========================================
# 5. Main Area (Game Area)
# ==========================================
st.title("Neural Odyssey 🌌")
st.subheader("Stage 5: 고전 vs 현대 (Classic ML vs Modern DL)")

if not st.session_state.get('trained', False) and not train_clicked:
    st.markdown("<div class='nexus-dialogue'><b>Nexus</b>: 이제 64개의 픽셀을 한 번에 읽어내서 숫자를 맞혀야 해요. 과거 통계학자들의 지혜(Classic ML)와 Keras가 제공하는 새로운 힘(Modern DL) 중 무엇이 더 뛰어날까요? 좌측의 학습 버튼을 눌러보세요!</div>", unsafe_allow_html=True)

# ==========================================
# 6. Training Logic (Both Models)
# ==========================================
if train_clicked:
    st.session_state['trained'] = False
    st.markdown("<div class='nexus-dialogue'><b>Nexus</b>: 두 세계의 힘을 비교해보고 있습니다... 결과가 어떻게 나올지 너무 궁금해요!</div>", unsafe_allow_html=True)
    
    col_ml, col_dl = st.columns(2)
    ml_container = col_ml.empty()
    dl_container = col_dl.empty()
    
    # 6.1 Classic ML (Logistic Regression)
    with ml_container.container():
        st.markdown("### 🏛️ Classic ML: Logistic Regression")
        ml_progress = st.progress(0, text="로지스틱 회귀 학습 중...")
        ml_model = LogisticRegression(solver='lbfgs', max_iter=epochs, random_state=42)
        
        # UX 애니메이션
        for i in range(10):
            time.sleep(0.05)
            ml_progress.progress((i+1)*10, text="로지스틱 회귀 최적화 진행 중...")
            
        ml_model.fit(X_train, y_train)
        ml_progress.empty()
        
        ml_preds = ml_model.predict(X_test)
        st.session_state['ml_acc'] = accuracy_score(y_test, ml_preds)
        st.session_state['ml_cm'] = confusion_matrix(y_test, ml_preds)
        
    # 6.2 Modern DL (Keras Single Dense Layer)
    with dl_container.container():
        st.markdown("### 🚀 Modern DL: Single Dense Layer")
        dl_progress = st.progress(0, text="Keras 단층 신경망 로딩 중...")
        
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.callbacks import Callback
        from tensorflow.keras.optimizers import Adam
        
        dl_model = Sequential([
            Dense(10, activation='softmax', input_shape=(64,))
        ])
        
        # LR 모델과 유사한 수렴 속도를 위해 Adam lr=0.01 적용
        dl_model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        class StreamlitCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                dl_progress.progress(progress, text=f"DL 최적화 진행 중... Epoch {epoch+1}/{epochs}")
                
        dl_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), callbacks=[StreamlitCallback()], verbose=0)
        dl_progress.empty()
        
        dl_loss, dl_acc = dl_model.evaluate(X_test, y_test, verbose=0)
        st.session_state['dl_acc'] = dl_acc
        
        dl_preds = np.argmax(dl_model.predict(X_test, verbose=0), axis=1)
        st.session_state['dl_cm'] = confusion_matrix(y_test, dl_preds)
        
    st.session_state['trained'] = True
    st.rerun()

# ==========================================
# 7. Render Results & Room of Truth
# ==========================================
if st.session_state.get('trained', False):
    st.markdown("<div class='nexus-dialogue'><b>Nexus</b>: 잠깐... 두 모델의 성능과 오차 행렬 분포가 놀라울 정도로 똑같아요! 밑에 있는 진실의 방(Room of Truth)을 열어서 수식을 비교해 보세요!</div>", unsafe_allow_html=True)
    
    col_ml, col_dl = st.columns(2)
    
    with col_ml:
        st.markdown("### 🏛️ Classic ML: Logistic Regression")
        st.success(f"**ML Test Accuracy:** {st.session_state['ml_acc']*100:.2f}%")
        
        fig_ml = px.imshow(st.session_state['ml_cm'], text_auto=True, color_continuous_scale='Blues', title="ML Confusion Matrix")
        fig_ml.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'))
        st.plotly_chart(fig_ml, use_container_width=True)
        
    with col_dl:
        st.markdown("### 🚀 Modern DL: Single Dense Layer")
        st.success(f"**DL Test Accuracy:** {st.session_state['dl_acc']*100:.2f}%")
        
        fig_dl = px.imshow(st.session_state['dl_cm'], text_auto=True, color_continuous_scale='Purples', title="DL Confusion Matrix")
        fig_dl.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'))
        st.plotly_chart(fig_dl, use_container_width=True)
        
    # Room of Truth (진실의 방)
    st.markdown("---")
    with st.expander("👁️ 진실의 방 (Room of Truth) 열기", expanded=True):
        st.markdown("### 두 모델은 왜 똑같은 결과를 냈을까요?")
        st.write("사실 **다중 클래스 로지스틱 회귀(Multinomial Logistic Regression)**와 **은닉층 없이 Softmax 활성화 함수만 거치는 단일 신경망(Single Dense Layer)**은 수학적으로 완전히 동일한 알고리즘입니다.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info("**로지스틱 회귀 수식 (고전)**")
            st.latex(r"P(y=k | x) = \frac{e^{w_k^T x + b_k}}{\sum_{j} e^{w_j^T x + b_j}}")
        with c2:
            st.info("**Keras 단층 신경망 + Softmax 수식 (현대)**")
            st.latex(r"\hat{y} = \text{Softmax}(W \cdot X + b)")
            
        st.success("결국 이름과 라이브러리만 다를 뿐, 딥러닝 인공신경망의 가장 기본이 되는 층(Dense)은 통계학자들이 수십 년 전부터 사용해오던 회귀 모델과 완벽하게 일치합니다. **딥러닝은 이러한 기본 모듈들을 엄청나게 깊게(Deep) 연결하고 GPU로 최적화하여 현대의 기적을 만들어내는 조립식 레고 블록과 같습니다!**")
        st.balloons()
