import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import io
import os

# Suppress TensorFlow logging to keep terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================


st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    h1, h2, h3 { color: #ff7b72; text-shadow: 0 0 10px #ff7b72; }
    .nexus-dialogue { 
        border-left: 5px solid #a5d6ff; 
        background-color: #161b22; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 20px;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. State & Data Initialization
# ==========================================
if 'stage4_init' not in st.session_state:
    st.session_state['stage4_init'] = True
    # 비선형 분포를 잘 나타내는 반달 모양 데이터셋
    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.session_state['data'] = (X, X_train, X_test, y_train, y_test)

X, X_train, X_test, y_train, y_test = st.session_state['data']

# ==========================================
# 3. UI Setup & Sidebar (Keras Workshop)
# ==========================================
with st.sidebar:
    st.title("🛠️ Keras Workshop")
    st.markdown("5단계 조립을 통해 완벽한 Keras 딥러닝 파이프라인을 완성하세요.")
    
    st.header("Step 1. Architecture")
    hidden_neurons = st.slider("은닉층 뉴런 수 (Hidden Neurons)", 4, 128, 32, step=4)
    activation = st.selectbox("활성화 함수 (Activation)", ["relu", "sigmoid", "tanh", "elu"])
    step1 = st.checkbox("✅ Step 1 완료 (모델 뼈대 잡기)")
    
    st.divider()
    
    st.header("Step 2. Compile")
    optimizer = st.selectbox("최적화 알고리즘 (Optimizer)", ["adam", "rmsprop", "sgd"])
    loss = st.selectbox("손실 함수 (Loss Function)", ["binary_crossentropy", "mse"])
    step2 = st.checkbox("✅ Step 2 완료 (학습 방법 설정)", disabled=not step1)
    
    st.divider()
    
    st.header("Step 3. Fit (Train)")
    epochs = st.slider("에포크 (Epochs)", 10, 200, 100, step=10)
    batch_size = st.slider("배치 사이즈 (Batch Size)", 8, 64, 16, step=8)
    step3 = st.checkbox("✅ Step 3 완료 (학습 루프 설정)", disabled=not step2)
    
    st.divider()
    
    st.header("Step 4. Evaluate")
    st.markdown("테스트 데이터로 모델을 객관적으로 평가합니다.")
    step4 = st.checkbox("✅ Step 4 완료 (평가 로직 추가)", disabled=not step3)
    
    st.divider()
    
    st.header("Step 5. Summary")
    st.markdown("전체 모델의 파라미터와 구조 요약을 출력합니다.")
    step5 = st.checkbox("✅ Step 5 완료 (요약 출력)", disabled=not step4)

# ==========================================
# 4. Main Area: Nexus Dialogue & Code Generator
# ==========================================
st.title("Neural Odyssey 🌌")
st.subheader("Stage 4: 신경망 조립술 (The Keras 5-Step Craft)")

steps_completed = sum([step1, step2, step3, step4, step5])

nexus_messages = [
    "Nexus: Keras 공방에 오신 것을 환영합니다! 첫 번째 단계인 모델의 구조(Architecture)를 좌측에서 설계하고 체크박스를 눌러 조립을 시작하세요.",
    "Nexus: 훌륭해요! 모델의 뼈대가 생겼습니다. 이제 이 모델의 오차를 어떻게 줄일지 최적화 방법(Compile)을 정해주세요.",
    "Nexus: 컴파일 로직이 추가되었습니다! 모델이 데이터를 몇 번이나 반복해서 볼지, 한 번에 몇 개씩 묶어서 볼지(Fit) 설정해주세요.",
    "Nexus: 학습 루프가 완성되었습니다! 학습이 끝난 후 모델이 처음 보는 데이터에서 얼마나 잘하는지 평가(Evaluate)할 준비를 하세요.",
    "Nexus: 평가 로직 추가 완료! 마지막으로 모델의 요약 정보(Summary)를 확인할 준비를 마치면 제너레이터가 완성됩니다.",
    "Nexus: 완벽합니다! Keras 5단계 파이프라인이 모두 조립되었습니다. 아래의 <b>'🚀 실행 (Run Keras Model)'</b> 버튼을 눌러 코드를 생명체로 깨워주세요!"
]

st.markdown(f"<div class='nexus-dialogue'><b>{nexus_messages[steps_completed]}</b></div>", unsafe_allow_html=True)

# Code Generator String Builder
code_str = ""
if step1:
    code_str += f"""# Step 1: Architecture (모델 구조 정의)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense({hidden_neurons}, activation='{activation}', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])\n\n"""

if step2:
    code_str += f"""# Step 2: Compile (학습 방법 설정)
model.compile(optimizer='{optimizer}', 
              loss='{loss}', 
              metrics=['accuracy'])\n\n"""

if step3:
    code_str += f"""# Step 3: Fit (모델 학습)
history = model.fit(X_train, y_train, 
                    epochs={epochs}, 
                    batch_size={batch_size}, 
                    validation_data=(X_test, y_test),
                    verbose=0)\n\n"""

if step4:
    code_str += f"""# Step 4: Evaluate (모델 평가)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {{accuracy*100:.2f}}%")\n\n"""

if step5:
    code_str += f"""# Step 5: Summary (모델 요약)
model.summary()"""

if not step1:
    code_str = "# Keras 코드가 여기에 실시간으로 번역되어 조립됩니다. 좌측 사이드바에서 단계를 진행하세요.\n"

st.markdown("### 💻 실시간 Keras 파이썬 코드 제너레이터")
st.code(code_str, language='python')

# ==========================================
# 5. Training Backend Logic (Keras)
# ==========================================
if steps_completed == 5:
    st.markdown("---")
    run_clicked = st.button("🚀 실행 (Run Keras Model)", use_container_width=True)
    
    if run_clicked:
        # TensorFlow 로딩 시간 동안 사용자에게 피드백 제공
        with st.spinner("TensorFlow 백엔드를 초기화하고 학습을 시작합니다... (최초 실행 시 로딩 지연이 발생할 수 있습니다)"):
            
            # 지연 로딩(Lazy Loading)으로 앱 최초 실행 속도 최적화
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.callbacks import Callback
            
            # Step 1
            model = Sequential([
                Dense(hidden_neurons, activation=activation, input_shape=(2,)),
                Dense(1, activation='sigmoid')
            ])
            
            # Step 2
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            
            # Step 3 (Streamlit 프로그레스 바 연동)
            progress_bar = st.progress(0, text="Keras 모델 학습 준비 중...")
            
            class StreamlitCallback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress, text=f"Keras 모델 학습 중... Epoch {epoch+1}/{epochs} | Loss: {logs['loss']:.4f}")
                    
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                validation_data=(X_test, y_test), 
                                callbacks=[StreamlitCallback()], verbose=0)
                                
            progress_bar.empty()
            
            # Step 4
            eval_loss, eval_acc = model.evaluate(X_test, y_test, verbose=0)
            
            # Step 5 (Summary 캡처)
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            summary_string = stream.getvalue()
            
            # 시각화 (Plots)
            st.markdown("### 📊 학습 결과 및 모델 분석")
            col1, col2 = st.columns(2)
            
            with col1:
                # Decision Boundary (Contour)
                x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
                y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
                grid = np.c_[xx.ravel(), yy.ravel()]
                
                preds = model.predict(grid, verbose=0)
                Z = preds.reshape(xx.shape)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.05), y=np.arange(y_min, y_max, 0.05), 
                                          z=Z, colorscale='RdBu', opacity=0.5, showscale=False))
                                          
                fig1.add_trace(go.Scatter(x=X_test[y_test==0, 0], y=X_test[y_test==0, 1], mode='markers',
                                          marker=dict(color='#3182bd', size=12, line=dict(color='white', width=1.5)), name='Test Class 0'))
                fig1.add_trace(go.Scatter(x=X_test[y_test==1, 0], y=X_test[y_test==1, 1], mode='markers',
                                          marker=dict(color='#e6550d', size=12, line=dict(color='white', width=1.5)), name='Test Class 1'))
                                          
                fig1.update_layout(title="결정 경계 (테스트 데이터 오버레이)", height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'))
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                # Loss Curve
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss', line=dict(color='#ff7b72', width=2)))
                fig2.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Val Loss', line=dict(color='#79c0ff', width=2)))
                fig2.update_layout(title="Loss Curve (손실률 감소 추이)", height=400, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c9d1d9'))
                st.plotly_chart(fig2, use_container_width=True)
                
            # Summary Output
            with st.expander("📝 Model Summary 확인하기", expanded=True):
                st.code(summary_string, language='text')
                
            # Achievement Logic
            st.success(f"**최종 테스트 정확도 (Test Accuracy)**: {eval_acc*100:.2f}%")
            if eval_acc >= 0.90:
                st.markdown("---")
                st.balloons()
                st.success("🏆 **업적 해금: Keras Craftsman!**\n\n완벽하게 5단계 조립을 마스터하고 90% 이상의 정확도를 달성했습니다. 당신의 모델 설계는 아름답고 강력합니다.")
                if st.button("🚀 Stage 5로 이동 (마지막 단계)", use_container_width=True):
                    st.switch_page("DL001/DL06_game.py")
            else:
                st.warning("⚠️ 정확도가 90%에 미치지 못했습니다. 에포크를 늘리거나 뉴런 수, Optimizer를 변경하여 모델을 더 강하게 만들어 보세요!")
