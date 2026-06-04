import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_digits
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="AI 조련사", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
    }
    h1, h2, h3 { color: #ffcc00; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI 조련사: 순전파와 역전파 훈련 게임")
st.markdown("가장 기초적인 딥러닝 연산을 시각적으로 체험합니다. **순전파**로 결과를 예측하고, **역전파**로 가중치(신경망의 연결 강도)를 조절하여 오차를 줄여보세요!")

@st.cache_resource
def get_digits():
    return load_digits()

digits = get_digits()

def get_random_problem():
    idx = np.random.randint(len(digits.data))
    img = digits.data[idx] / 16.0 
    label = digits.target[idx]
    return img.reshape(64, 1), label

# 상태 초기화
if 'w1' not in st.session_state:
    np.random.seed(42) # 시드 고정으로 첫 화면 동일하게
    st.session_state.w1 = np.random.randn(12, 64) * 0.5 
    st.session_state.b1 = np.zeros((12, 1))
    st.session_state.w2 = np.random.randn(10, 12) * 0.5
    st.session_state.b2 = np.zeros((10, 1))
    
    img, label = get_random_problem()
    st.session_state.img = img
    st.session_state.label = label
    st.session_state.epoch = 0
    st.session_state.loss = 2.3
    st.session_state.preds = np.ones(10) / 10.0
    st.session_state.state = 'ready'
    st.session_state.msg = ""

def forward_pass(img, label):
    z1 = np.dot(st.session_state.w1, img) + st.session_state.b1
    a1 = np.maximum(0, z1) # ReLU 활성화 함수
    z2 = np.dot(st.session_state.w2, a1) + st.session_state.b2
    exp_z = np.exp(z2 - np.max(z2))
    a2 = exp_z / np.sum(exp_z) # Softmax
    
    y_true = np.zeros((10, 1))
    y_true[label] = 1.0
    loss = -np.sum(y_true * np.log(a2 + 1e-8)) # Cross-Entropy Loss
    
    return a1, a2, loss

def backward_pass(img, label, a1, a2):
    y_true = np.zeros((10, 1))
    y_true[label] = 1.0
    
    dz2 = a2 - y_true
    dw2 = np.dot(dz2, a1.T)
    db2 = dz2
    
    da1 = np.dot(st.session_state.w2.T, dz2)
    z1 = np.dot(st.session_state.w1, img) + st.session_state.b1
    dz1 = da1 * (z1 > 0)
    dw1 = np.dot(dz1, img.T)
    db1 = dz1
    
    lr = 0.05
    st.session_state.w1 -= lr * dw1
    st.session_state.b1 -= lr * db1
    st.session_state.w2 -= lr * dw2
    st.session_state.b2 -= lr * db2

# Plotly 네트워크 시각화 함수
def draw_network_fast(W1, W2, img=None, a1=None, a2=None, state='ready'):
    fig = go.Figure()
    
    in_x = np.zeros(64)
    in_y = np.linspace(-15, 15, 64)
    hid_x = np.ones(12) * 1
    hid_y = np.linspace(-8, 8, 12)
    out_x = np.ones(10) * 2
    out_y = np.linspace(-6, 6, 10)
    
    def add_edges_to_bins(W, start_x, start_y, end_x, end_y, max_edges=None):
        bins = {'blue_1': [], 'blue_3': [], 'blue_5': [],
                'red_1': [], 'red_3': [], 'red_5': []}
                
        if max_edges:
            flat = np.abs(W).flatten()
            if len(flat) > max_edges:
                threshold = np.sort(flat)[-max_edges]
            else:
                threshold = 0
        else:
            threshold = 0
            
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = W[i, j]
                if abs(w) >= threshold and abs(w) > 0.1:
                    color = 'blue' if w > 0 else 'red'
                    thick = abs(w) * 3
                    if thick < 1.5: b = '1'
                    elif thick < 3.5: b = '3'
                    else: b = '5'
                    
                    b_key = f"{color}_{b}"
                    bins[b_key].extend([start_x[j], end_x[i], None])
                    bins[b_key].extend([start_y[j], end_y[i], None])
        return bins
    
    b1 = add_edges_to_bins(W1, in_x, in_y, hid_x, hid_y, max_edges=40)
    b2 = add_edges_to_bins(W2, hid_x, hid_y, out_x, out_y)
    
    all_bins = {'blue_1': [], 'blue_3': [], 'blue_5': [], 'red_1': [], 'red_3': [], 'red_5': []}
    for k in all_bins.keys():
        x_list = []
        y_list = []
        if len(b1[k]) > 0:
            x_list.extend(b1[k][::2])
            y_list.extend(b1[k][1::2])
        if len(b2[k]) > 0:
            x_list.extend(b2[k][::2])
            y_list.extend(b2[k][1::2])
            
        if x_list:
            c = 'rgba(0, 200, 255, 0.4)' if 'blue' in k else 'rgba(255, 0, 85, 0.4)'
            w = int(k.split('_')[1])
            fig.add_trace(go.Scatter(x=x_list, y=y_list, mode='lines', line=dict(color=c, width=w), hoverinfo='none', showlegend=False))

    # 노드 색상과 크기를 활성화 또는 에러(Error)에 따라 동적으로 변경
    if state == 'forwarded' and img is not None:
        in_color = [f'rgba(255, 255, 255, {min(1.0, max(0.1, float(v[0])))})' for v in img]
        hid_color = [f'rgba(255, 204, 0, {min(1.0, max(0.2, float(v[0])))})' for v in a1]
        out_color = [f'rgba(0, 255, 204, {min(1.0, max(0.2, float(v[0])))})' for v in a2]
        in_size = [4 + float(v[0])*5 for v in img]
        hid_size = [8 + float(v[0])*12 for v in a1]
        out_size = [8 + float(v[0])*15 for v in a2]
        
    elif state.startswith('backward'):
        # 역전파 시 Error 방향(양/음)에 따라 붉은색/푸른색 경고 렌더링
        def err_color(val):
            v = float(val[0])
            if v > 0: return f'rgba(255, 0, 85, {min(1.0, max(0.2, v*3))})'
            else: return f'rgba(0, 200, 255, {min(1.0, max(0.2, abs(v)*3))})'
        def err_size(val):
            return 8 + min(15, abs(float(val[0]))*25)
            
        in_color = [f'rgba(255, 255, 255, {min(1.0, max(0.1, float(v[0])))})' for v in img]
        in_size = [4 + float(v[0])*5 for v in img]
        
        if state in ['backward_step1', 'backward_step2']:
            hid_color = [f'rgba(255, 204, 0, {min(1.0, max(0.2, float(v[0])))})' for v in a1] 
            hid_size = [8 + float(v[0])*12 for v in a1]
        else: # step 3, 4
            hid_color = [err_color(v) for v in a1] # a1 위치에 dz1이 들어옴
            hid_size = [err_size(v) for v in a1]
            
        out_color = [err_color(v) for v in a2] # a2 위치에 dz2가 들어옴
        out_size = [err_size(v) for v in a2]
        
    else:
        in_color = '#333333'
        hid_color = '#554400'
        out_color = '#004433'
        in_size = 4
        hid_size = 12
        out_size = 12

    fig.add_trace(go.Scatter(x=in_x, y=in_y, mode='markers', marker=dict(size=in_size, color=in_color, line=dict(width=1, color='#555')), hoverinfo='none', showlegend=False))
    fig.add_trace(go.Scatter(x=hid_x, y=hid_y, mode='markers', marker=dict(size=hid_size, color=hid_color, line=dict(width=1, color='#aaa')), hoverinfo='none', showlegend=False))
    fig.add_trace(go.Scatter(x=out_x, y=out_y, mode='markers', marker=dict(size=out_size, color=out_color, line=dict(width=1, color='#aaa')), hoverinfo='none', showlegend=False))
    
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=0, t=30, b=0), height=450, title="가중치(Weight) 네트워크 시각화")
    return fig

st.markdown("---")
# UI 레이아웃 분할
col1, col2, col3 = st.columns([1, 2, 1])

# Left Column
with col1:
    st.subheader("1. 문제 출제")
    if st.button("🔄 다른 문제 뽑기", use_container_width=True):
        img, label = get_random_problem()
        st.session_state.img = img
        st.session_state.label = label
        st.session_state.state = 'ready'
        st.rerun()
        
    img_2d = st.session_state.img.reshape(8, 8)
    fig_img = go.Figure(data=go.Heatmap(z=img_2d, colorscale='gray_r', showscale=False))
    fig_img.update_layout(width=250, height=250, margin=dict(l=20, r=20, t=20, b=20), xaxis=dict(visible=False), yaxis=dict(visible=False, autorange='reversed'), template="plotly_dark")
    st.plotly_chart(fig_img, use_container_width=True)
    st.markdown(f"### 정답: **{st.session_state.label}**")

# Center Column
with col2:
    st.subheader("2. 신경망 시각화 및 제어")
    c2_1, c2_2 = st.columns(2)
    
    info_placeholder = st.empty()
    graph_placeholder = st.empty()
    
    if c2_1.button("▶ 순전파 (예측하기)", use_container_width=True):
        if st.session_state.state.startswith('backward'):
            st.session_state.msg = "warning|현재 역전파 진행 중입니다. 끝까지 완료 후 다시 시도하세요!"
        else:
            a1, a2, loss = forward_pass(st.session_state.img, st.session_state.label)
            st.session_state.a1 = a1
            st.session_state.a2 = a2
            st.session_state.loss = loss
            st.session_state.preds = a2.flatten()
            st.session_state.state = 'forwarded'
            st.session_state.msg = ""
            
    # 버튼 텍스트 동적 할당
    if st.session_state.state == 'forwarded':
        btn_label = "◀ 1단계: 출력층 오차 확인"
    elif st.session_state.state == 'backward_step1':
        btn_label = "◀ 2단계: W2 파이프 수정"
    elif st.session_state.state == 'backward_step2':
        btn_label = "◀ 3단계: 은닉층 오차 확인"
    elif st.session_state.state == 'backward_step3':
        btn_label = "◀ 4단계: W1 파이프 수정"
    else:
        btn_label = "◀ 역전파 시작"
        
    if c2_2.button(btn_label, use_container_width=True):
        if st.session_state.state == 'ready' or st.session_state.state == 'ready_after_backprop':
            st.session_state.msg = "warning|먼저 순전파를 실행하여 오차를 계산하세요!"
            
        elif st.session_state.state == 'forwarded':
            y_true = np.zeros((10, 1))
            y_true[st.session_state.label] = 1.0
            st.session_state.dz2 = st.session_state.a2 - y_true
            st.session_state.state = 'backward_step1'
            st.session_state.msg = ""
            st.rerun()
            
        elif st.session_state.state == 'backward_step1':
            dw2 = np.dot(st.session_state.dz2, st.session_state.a1.T)
            st.session_state.w2 -= 0.05 * dw2
            st.session_state.b2 -= 0.05 * st.session_state.dz2
            st.session_state.state = 'backward_step2'
            st.session_state.msg = ""
            st.rerun()
            
        elif st.session_state.state == 'backward_step2':
            da1 = np.dot(st.session_state.w2.T, st.session_state.dz2)
            z1 = np.dot(st.session_state.w1, st.session_state.img) + st.session_state.b1
            st.session_state.dz1 = da1 * (z1 > 0)
            st.session_state.state = 'backward_step3'
            st.session_state.msg = ""
            st.rerun()
            
        elif st.session_state.state == 'backward_step3':
            dw1 = np.dot(st.session_state.dz1, st.session_state.img.T)
            st.session_state.w1 -= 0.05 * dw1
            st.session_state.b1 -= 0.05 * st.session_state.dz1
            st.session_state.epoch += 1
            st.session_state.state = 'ready_after_backprop'
            st.session_state.msg = ""
            st.rerun()

    # 클릭을 통해 진척되는 렌더링 시퀀스
    if st.session_state.state == 'forwarded':
        info_placeholder.info("🟢 **순전파(Forward) 완료!**\n\n1. 입력 이미지가 왼쪽 노드들에 불을 켭니다.\n2. 가중치(파이프)를 타고 신호가 이동하며, 중앙 **은닉층 노드**들이 활성화(노란색 빛)됩니다.\n3. 오른쪽 **출력층 노드**에서 최종 확률이 계산되며, 정답과의 **오차(Loss)**가 발생합니다.\n\n*이제 '◀ 1단계' 버튼을 눌러 역전파를 차례대로 진행하세요!*")
        fig_net = draw_network_fast(st.session_state.w1, st.session_state.w2, st.session_state.img, st.session_state.a1, st.session_state.a2, 'forwarded')
        graph_placeholder.plotly_chart(fig_net, use_container_width=True)
        
    elif st.session_state.state == 'backward_step1':
        info_placeholder.error("🔴 **[역전파 1단계] 출력층 오차 확인!**\n\n오른쪽 출력층에서 실제 정답과 예측값의 차이(오차)가 계산되었습니다. (에러 크기에 따라 노드가 붉은/푸른 빛을 냅니다)\n\n*다음 단계를 눌러 가중치 파이프를 수정하세요!*")
        fig_net = draw_network_fast(st.session_state.w1, st.session_state.w2, st.session_state.img, st.session_state.a1, st.session_state.dz2, 'backward_step1')
        graph_placeholder.plotly_chart(fig_net, use_container_width=True)
        
    elif st.session_state.state == 'backward_step2':
        info_placeholder.warning("🟠 **[역전파 2단계] W2 가중치 파이프 수정!**\n\n발생한 오차를 거슬러 올라가며, 은닉층 ➔ 출력층 사이의 연결선(W2) 굵기를 즉각 조절했습니다.\n\n*다음 단계를 눌러 오차를 은닉층으로 넘기세요!*")
        fig_net = draw_network_fast(st.session_state.w1, st.session_state.w2, st.session_state.img, st.session_state.a1, st.session_state.dz2, 'backward_step2')
        graph_placeholder.plotly_chart(fig_net, use_container_width=True)
        
    elif st.session_state.state == 'backward_step3':
        info_placeholder.error("🔴 **[역전파 3단계] 은닉층으로 오차 전파!**\n\n수정된 파이프를 타고 역류하여, 중앙의 은닉층 노드들에 오차 책임(Error)이 각각 분배되었습니다.\n\n*마지막 단계를 눌러 남은 파이프를 수정하세요!*")
        fig_net = draw_network_fast(st.session_state.w1, st.session_state.w2, st.session_state.img, st.session_state.dz1, st.session_state.dz2, 'backward_step3')
        graph_placeholder.plotly_chart(fig_net, use_container_width=True)
        
    elif st.session_state.state == 'ready_after_backprop' and st.session_state.epoch > 0:
        info_placeholder.success("✅ **역전파(Backward) 모든 단계 완료!**\n\n오차가 왼쪽으로 4단계에 걸쳐 거꾸로 전달되며, 모든 가중치 파이프들의 굵기가 실시간으로 수정되었습니다. (좌측에서 새로운 데이터로 다시 시도해보세요!)")
        fig_net = draw_network_fast(st.session_state.w1, st.session_state.w2)
        graph_placeholder.plotly_chart(fig_net, use_container_width=True)
        
    elif st.session_state.state == 'ready':
        info_placeholder.info("👆 **[▶ 순전파 예측하기]** 버튼을 눌러 인공지능의 뇌에 신호(빛)를 통과시켜 보세요.")
        fig_net = draw_network_fast(st.session_state.w1, st.session_state.w2)
        graph_placeholder.plotly_chart(fig_net, use_container_width=True)

    if st.session_state.msg:
        msg_type, msg_text = st.session_state.msg.split("|")
        if msg_type == "warning": st.warning(msg_text)

# Right Column
with col3:
    st.subheader("3. 결과 및 상태")
    st.metric("진행된 학습 횟수 (Epoch)", st.session_state.epoch)
    st.metric("현재 오차 (Loss)", f"{st.session_state.loss:.4f}")
    
    if st.button("⚡ 자동 10번 반복 학습", use_container_width=True):
        for _ in range(10):
            a1, a2, loss = forward_pass(st.session_state.img, st.session_state.label)
            backward_pass(st.session_state.img, st.session_state.label, a1, a2)
            st.session_state.epoch += 1
        
        # 마지막 결과를 화면에 반영하기 위해 한번 더 순전파
        a1, a2, loss = forward_pass(st.session_state.img, st.session_state.label)
        st.session_state.a1 = a1
        st.session_state.a2 = a2
        st.session_state.loss = loss
        st.session_state.preds = a2.flatten()
        st.session_state.state = 'forwarded'
        st.session_state.msg = ""
        st.rerun() # UI 리프레시
    
    # 예측 결과 Bar 차트
    fig_bar = go.Figure(data=go.Bar(x=[str(i) for i in range(10)], y=st.session_state.preds, marker_color='#00ffcc'))
    colors = ['#00ffcc'] * 10
    colors[st.session_state.label] = '#ff0055' # 정답 클래스 강조
    fig_bar.update_traces(marker_color=colors)
    fig_bar.update_layout(title="숫자별 예측 확률", yaxis=dict(range=[0, 1]), height=250, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.warning("💡 **목표:** 빨간색 막대(정답)가 1.0(100%)에 가깝게 제일 높아지도록 학습시켜보세요!")
