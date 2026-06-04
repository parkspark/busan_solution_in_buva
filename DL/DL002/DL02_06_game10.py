import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import os

# TensorFlow 로그 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- UI 기본 설정 ---
st.set_page_config(page_title="Neural Odyssey: Stage 10", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
    }
    h1, h2, h3 {
        color: #ffcc00;
        text-shadow: 0 0 8px rgba(255, 204, 0, 0.5);
    }
    .nexus-dialogue {
        background: linear-gradient(90deg, rgba(255, 204, 0, 0.1) 0%, rgba(0, 255, 204, 0.05) 100%);
        border-left: 5px solid #ffcc00;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.1em;
        box-shadow: 0 4px 15px rgba(255, 204, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 파라미터 계산 유틸리티 ---
def count_params(in_dim, layers, nodes, out_dim):
    if layers == 0: return in_dim * out_dim + out_dim
    p = (in_dim * nodes) + nodes
    p += (layers - 1) * ((nodes * nodes) + nodes)
    p += (nodes * out_dim) + out_dim
    return p

# --- 데이터 셋업 (캐싱) ---
@st.cache_data
def get_data():
    digits = load_digits()
    scaler = StandardScaler()
    X = scaler.fit_transform(digits.data)
    y = digits.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

# --- Keras 모델 빌더 ---
def build_model(num_layers, nodes):
    # 동일한 조건 비교를 위해 시드 고정
    initializer = tf.keras.initializers.GlorotUniform(seed=42)
    inputs = tf.keras.Input(shape=(64,))
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(nodes, activation='relu', kernel_initializer=initializer)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer)(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Session State ---
if 'prev_params' not in st.session_state:
    st.session_state.prev_params = 0
if 'cleared' not in st.session_state:
    st.session_state.cleared = False
if 'stars' not in st.session_state:
    st.session_state.stars = 0

# --- 사이드바 (Deep Architecture Studio) ---
st.sidebar.title("Deep Architecture Studio")
st.sidebar.markdown("신경망의 깊이(Layer)와 너비(Nodes)를 조립하세요.")

num_layers = st.sidebar.slider("은닉층 (Hidden Layers) 개수", 1, 6, 3)
nodes = st.sidebar.selectbox("각 층당 뉴런 수 (Nodes)", [16, 64, 144, 256], index=1, format_func=lambda x: f"{x}개 ({int(np.sqrt(x))}x{int(np.sqrt(x))} 해상도)")

# 실시간 파라미터 카운터 (델타 애니메이션)
current_params = count_params(64, num_layers, nodes, 10)
delta = current_params - st.session_state.prev_params

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ 모델 무게 측정기")
st.sidebar.metric("총 파라미터 수 (Total Parameters)", f"{current_params:,} 개", delta=f"{delta:,}" if delta != 0 else None, delta_color="inverse", help="파라미터가 너무 많으면 연산이 느려지고 과적합(Overfitting)에 빠집니다.")
st.session_state.prev_params = current_params

st.sidebar.markdown("---")
train_btn = st.sidebar.button("🚀 Nexus 심층 학습 시작")

st.sidebar.markdown("---")
st.sidebar.subheader("🌟 미션 달성 조건 (별 3개)")
st.sidebar.markdown("- ★ : 3층 이상 모델로 1층 모델(Shallow) 성능 압도하기\n- ★★ : Validation 정확도 **95%** 이상 달성\n- ★★★ : 총 파라미터 **20,000개 이하**로 95% 달성 (효율성 극대화)")


# --- 메인 영역 (Game Area & Nexus Insight) ---
st.title("Neural Odyssey - Stage 10: 깊이의 힘 (The Power of Depth)")

# 상단: Nexus 대사
nexus_ph = st.empty()
if current_params > 50000:
    nexus_ph.markdown('<div class="nexus-dialogue" style="border-left-color:#ff0055; color:#ff0055;">🤖 <b>Nexus:</b> 머리가 너무 무거워요...! 파라미터가 5만 개를 넘어가서 배운 것만 외우고 새로운 문제는 못 풀지도 몰라요 (과적합 위험)!</div>', unsafe_allow_html=True)
elif num_layers >= 4:
    nexus_ph.markdown('<div class="nexus-dialogue" style="border-left-color:#00ffcc; color:#00ffcc;">🤖 <b>Nexus:</b> 세상이 훨씬 더 깊고 입체적으로 보입니다! 중간층에서 픽셀들이 추상적인 패턴으로 완벽히 변환되고 있어요!</div>', unsafe_allow_html=True)
else:
    nexus_ph.markdown('<div class="nexus-dialogue">🤖 <b>Nexus:</b> 층을 쌓고 뉴런을 넓혀 제 뇌의 용량(파라미터)을 늘려주세요. 하지만 무조건 무겁게 만든다고 정답일까요? 효율적인 조율이 필요합니다.</div>', unsafe_allow_html=True)

# 중앙: 계층적 특징 시각화
st.subheader("👁️ 중간 표현 시각화 (Feature Representation Learning)")
st.markdown("입력 이미지 데이터가 신경망의 각 층(Layer)을 통과하며 어떻게 추상적인 활성화 맵(Activation Map)으로 변환되는지 관찰하세요. 깊은 층일수록 더 복잡한 특징을 요약합니다.")
feature_ph = st.empty()

st.markdown("---")
# 하단: 과적합 아레나
st.subheader("⚔️ 과적합 아레나 (Overfitting Arena)")
st.markdown("단층(1-Layer) 모델, 적정 깊이(3-Layer) 모델, 그리고 **당신이 설계한 N층 모델**의 훈련 과정을 실시간으로 비교합니다.")
col1, col2 = st.columns(2)
with col1:
    chart_acc_ph = st.empty()
with col2:
    chart_loss_ph = st.empty()

# --- 훈련 스크립트 ---
if train_btn:
    X_train, X_test, y_train, y_test = get_data()
    
    tf.keras.backend.clear_session()
    
    # 베이스라인 모델 생성 (비교용)
    m1 = build_model(1, 64)
    m3 = build_model(3, 64)
    # 유저 모델 생성
    mn = build_model(num_layers, nodes)
    
    epochs = 30
    hist = {'m1_acc':[], 'm1_val':[], 'm3_acc':[], 'm3_val':[], 'mn_acc':[], 'mn_val':[], 'mn_loss':[], 'mn_val_loss':[]}
            
    progress = st.progress(0)
    
    for epoch in range(1, epochs + 1):
        # 1에포크씩 동시 학습
        h1 = m1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, verbose=0)
        h3 = m3.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, verbose=0)
        hn = mn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, verbose=0)
        
        # 기록 저장
        hist['m1_acc'].append(h1.history['accuracy'][0])
        hist['m1_val'].append(h1.history['val_accuracy'][0])
        hist['m3_acc'].append(h3.history['accuracy'][0])
        hist['m3_val'].append(h3.history['val_accuracy'][0])
        hist['mn_acc'].append(hn.history['accuracy'][0])
        hist['mn_val'].append(hn.history['val_accuracy'][0])
        hist['mn_loss'].append(hn.history['loss'][0])
        hist['mn_val_loss'].append(hn.history['val_loss'][0])
        
        # UI 업데이트: 정확도 차트
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(y=hist['m1_val'], name='1-Layer (Val)', line=dict(dash='dash', color='gray')))
        fig_acc.add_trace(go.Scatter(y=hist['m3_val'], name='3-Layer (Val)', line=dict(dash='dot', color='lightblue')))
        fig_acc.add_trace(go.Scatter(y=hist['mn_acc'], name=f'My {num_layers}-Layer (Train)', line=dict(color='#ffcc00')))
        fig_acc.add_trace(go.Scatter(y=hist['mn_val'], name=f'My {num_layers}-Layer (Val)', line=dict(color='#00ffcc', width=3)))
        fig_acc.update_layout(title="Accuracy (성능 대결)", template="plotly_dark", height=350, margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        chart_acc_ph.plotly_chart(fig_acc, use_container_width=True)
        
        # UI 업데이트: 손실 차트 (과적합 관찰)
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=hist['mn_loss'], name=f'Train Loss (학습 손실)', line=dict(color='#ffcc00')))
        fig_loss.add_trace(go.Scatter(y=hist['mn_val_loss'], name=f'Validation Loss (검증 손실)', line=dict(color='#00ffcc', width=3)))
        fig_loss.update_layout(title="My Model Loss (과적합 탐지기)", template="plotly_dark", height=350, margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        chart_loss_ph.plotly_chart(fig_loss, use_container_width=True)
        
        progress.progress(epoch / epochs)
        time.sleep(0.01)
        
    progress.empty()
        
    # --- Feature Visualization (학습 완료 후 중간 표현 추출) ---
    dense_layers = [l for l in mn.layers if isinstance(l, tf.keras.layers.Dense)][:-1] # 출력층 제외
    if dense_layers:
        feat_model = tf.keras.Model(inputs=mn.input, outputs=[l.output for l in dense_layers])
        # 테스트 이미지 1개로 특성 맵 추출
        acts = feat_model.predict(X_test[0:1], verbose=0)
        if not isinstance(acts, list): acts = [acts]
        
        with feature_ph.container():
            num_cols = len(acts) + 1
            f_cols = st.columns(num_cols)
            
            # 입력 이미지 플로팅
            img_in = X_test[0].reshape(8, 8)
            fig_in = go.Figure(data=go.Heatmap(z=img_in, colorscale='gray', showscale=False))
            fig_in.update_layout(title="Input (8x8)", height=180, margin=dict(l=10,r=10,t=30,b=10), xaxis=dict(visible=False), yaxis=dict(visible=False, autorange='reversed'))
            f_cols[0].plotly_chart(fig_in, use_container_width=True)
            
            # 은닉층 활성화 맵 플로팅 (1D Dense를 2D 해상도로 변환)
            side = int(np.sqrt(nodes))
            for i, act in enumerate(acts):
                act_map = act[0].reshape(side, side)
                fig_act = go.Figure(data=go.Heatmap(z=act_map, colorscale='inferno', showscale=False))
                fig_act.update_layout(title=f"Layer {i+1} ({side}x{side})", height=180, margin=dict(l=10,r=10,t=30,b=10), xaxis=dict(visible=False), yaxis=dict(visible=False, autorange='reversed'))
                f_cols[i+1].plotly_chart(fig_act, use_container_width=True)

    # --- 업적/별점 평가 로직 ---
    final_val_acc = hist['mn_val'][-1]
    m1_val_acc = hist['m1_val'][-1]
    
    st.session_state.cleared = True
    stars = 0
    if num_layers >= 3 and final_val_acc >= m1_val_acc: stars += 1
    if final_val_acc >= 0.95: stars += 1 
    if current_params <= 20000 and final_val_acc >= 0.95: stars += 1
    
    st.session_state.stars = stars

# --- 클리어 메시지 출력 ---
if st.session_state.cleared:
    stars = st.session_state.stars
    st.balloons()
    st.success(f"### 🏆 쾌거! 심층 설계의 마스터 달성! (획득 별점: {'★'*stars}{'☆'*(3-stars)})")
    if stars == 3:
        st.info("완벽합니다! 파라미터를 20,000개 이하로 최소화하면서도 모델의 깊이를 층층이 활용해 극강의 효율로 패턴을 학습해냈습니다. 이것이 바로 딥러닝(Deep Learning)의 진짜 매력입니다!")
    else:
        st.info("별 3개를 달성하려면, [파라미터 20,000개 이하] + [Validation 정확도 95% 이상] + [3층 이상] 조건을 모두 만족시켜보세요. 팁: 노드 수를 줄이고 깊이를 유지해보세요!")
