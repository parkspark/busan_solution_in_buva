import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import altair as alt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Vanishing Gradient Simulator", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e0e6ed;
    }
    h1, h2, h3 { color: #ffcc00; }
</style>
""", unsafe_allow_html=True)

st.title("📉 기울기 소멸(Vanishing Gradient) 시뮬레이터")
st.markdown("""
이 시뮬레이터는 신경망이 깊어질수록 뒤(출력층)에서 발생한 오차 신호(Gradient)가 앞(입력층)으로 거꾸로 전달될 때 **신호의 크기가 어떻게 변화하는지** 보여줍니다. 좌측에서 **은닉층의 개수**와 **활성화 함수**를 선택하고 역전파를 한 단계씩 실행해보세요!
""")

with st.expander("📖 **[필독] 기울기 소멸 현상이란 무엇인가요? (개념 설명)**", expanded=True):
    st.markdown("""
    인공지능은 **역전파(Backpropagation)**를 통해 출력층에서 발생한 오차를 거꾸로 흘려보내며 각 층의 가중치(파이프 굵기)를 수정합니다. 이 때 **연쇄 법칙(Chain Rule)**에 의해, 층을 하나 거슬러 올라갈 때마다 **해당 층에 쓰인 '활성화 함수(Activation)의 미분값'이 계속해서 곱해집니다.**

    - ❌ **Sigmoid 함수의 치명적 단점**: 
      Sigmoid 함수는 미분(기울기)의 최대값이 겨우 **0.25**입니다. 따라서 오차 신호가 층을 하나 지날 때마다 기존 크기에 0.25배(또는 그 이하)씩 계속 곱해집니다. ($1.0 \\times 0.25 \\times 0.25 \\times 0.25...$)
      결국 입력층에 가까운 앞쪽 은닉층들은 오차를 거의 전달받지 못해 **가중치 수정이 멈춰버리게** 되는데, 이를 **'기울기 소멸(Vanishing Gradient) 현상'**이라고 부릅니다. 과거 딥러닝이 암흑기를 겪었던 결정적인 원인입니다.

    - ✅ **ReLU 함수의 구원**: 
      ReLU 함수는 입력이 양수일 때 **미분값이 항상 1.0**입니다. 1은 아무리 여러 번 곱해도 1이므로, 층이 수십 개로 깊어져도 오차 신호가 소멸하지 않고 입력층까지 원래 크기 그대로 뚜렷하게 전달됩니다. 이 간단한 발견 덕분에 현대의 깊은 인공신경망(Deep Learning) 학습이 비로소 가능해졌습니다!
    """)

# State initialization and Reset Callback
if 'current_step' not in st.session_state:
    st.session_state.current_step = -1

def reset_step():
    st.session_state.current_step = -1

# Sidebar
st.sidebar.header("제어판 (Control Panel)")
num_layers = st.sidebar.slider("은닉층 개수 (Hidden Layers)", min_value=3, max_value=10, value=5, on_change=reset_step)
activation = st.sidebar.radio("활성화 함수 (Activation)", ["Sigmoid", "ReLU"], on_change=reset_step)

total_steps = num_layers + 1

# 동적 버튼 라벨
if st.session_state.current_step == -1:
    btn_label = "◀ 역전파 시작 (출력층 오차 확인)"
elif st.session_state.current_step < total_steps:
    btn_label = f"◀ 다음 층으로 오차 전파 ({st.session_state.current_step + 1}/{total_steps})"
else:
    btn_label = "🔄 처음부터 다시 해보기"

if st.sidebar.button(btn_label, use_container_width=True):
    if st.session_state.current_step == -1 or st.session_state.current_step == total_steps:
        st.session_state.current_step = 0
    else:
        st.session_state.current_step += 1

# Calculation Logic
def calculate_gradients(num_layers, activation):
    grads = [1.0] # Output layer gradient = 1.0
    multiplier = 0.25 if activation == "Sigmoid" else 1.0
    for _ in range(num_layers + 1):
        grads.append(grads[-1] * multiplier)
    return grads[::-1] # Reverse to match left-to-right [Input, H1, H2, ..., Output]

def draw_network(num_layers, current_step, grads):
    fig = go.Figure()
    
    total_nodes = num_layers + 2
    x_pos = np.linspace(0, 10, total_nodes)
    y_pos = np.zeros(total_nodes)
    
    active_idx = total_nodes - 1 - current_step if current_step >= 0 else total_nodes
    
    node_colors = []
    edge_colors = []
    edge_widths = []
    
    for i in range(total_nodes):
        if i > active_idx:
            g_val = grads[i]
            color_intensity = min(1.0, max(0.05, g_val))
            if activation == 'Sigmoid':
                c = f'rgba(255, 50, 80, {color_intensity})'
            else:
                c = f'rgba(50, 255, 150, {color_intensity})'
            node_colors.append(c)
        elif i == active_idx:
            node_colors.append('rgba(255, 255, 0, 1.0)')
        else:
            node_colors.append('rgba(60, 60, 60, 1.0)')
            
    for i in range(total_nodes - 1):
        if i >= active_idx:
            g_val = grads[i+1]
            thick = max(1, g_val * 8)
            opacity = min(1.0, max(0.1, g_val))
            if activation == 'Sigmoid':
                c = f'rgba(255, 50, 80, {opacity})'
            else:
                c = f'rgba(50, 255, 150, {opacity})'
            edge_colors.append(c)
            edge_widths.append(thick)
        else:
            edge_colors.append('rgba(60, 60, 60, 0.5)')
            edge_widths.append(2)

    for i in range(total_nodes - 1):
        fig.add_trace(go.Scatter(
            x=[x_pos[i], x_pos[i+1]], y=[y_pos[i], y_pos[i+1]],
            mode='lines',
            line=dict(color=edge_colors[i], width=edge_widths[i]),
            hoverinfo='none', showlegend=False
        ))
        
    labels = ['Input'] + [f'H{i}' for i in range(1, num_layers+1)] + ['Output']
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos, mode='markers+text',
        marker=dict(size=35, color=node_colors, line=dict(color='#cccccc', width=2)),
        text=labels, textposition="top center", textfont=dict(color='white', size=14),
        hoverinfo='text', showlegend=False
    ))
    
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-1, 1.5]),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20), height=300,
        title=f"신경망 시각화 ({activation} 적용) - 클릭할 때마다 왼쪽으로 오차 전파"
    )
    return fig

st.markdown("---")
# Main Layout
grads = calculate_gradients(num_layers, activation)
step = st.session_state.current_step

# 1. Update Network Plot
fig_net = draw_network(num_layers, step, grads)
st.plotly_chart(fig_net, use_container_width=True)

col1, col2 = st.columns([1, 1])

# 2. Update Bar Chart
total_nodes = num_layers + 2
active_idx = total_nodes - 1 - step if step >= 0 else total_nodes
show_grads = []
labels = ['Input'] + [f'H{i}' for i in range(1, num_layers+1)] + ['Output']

for i in range(total_nodes):
    if step >= 0 and i >= active_idx:
        show_grads.append(grads[i])
    else:
        show_grads.append(0.0)
    
df = pd.DataFrame({'Layer': labels, 'Gradient': show_grads})

chart = alt.Chart(df).mark_bar(color='#ff5555' if activation == 'Sigmoid' else '#55ff55').encode(
    x=alt.X('Layer:O', sort=labels, title='신경망 층 (오른쪽이 출력층)'),
    y=alt.Y('Gradient:Q', title='오차(Gradient) 크기', scale=alt.Scale(domain=[0, 1.0])),
    tooltip=['Layer', 'Gradient']
).properties(height=350, title=f"층별 오차 크기 변화량 ({activation})")

with col1:
    st.altair_chart(chart, use_container_width=True)

# 3. Text & Metric Feedback
with col2:
    if step == -1:
        st.info("👈 좌측 제어판에서 설정을 마치고 **'▶ 역전파 시작'** 버튼을 클릭하여 시뮬레이션을 한 단계씩 진행하세요.")
    elif step < total_steps:
        current_g = grads[max(active_idx, 0)]
        layer_name = labels[active_idx]
        st.info(f"""
        ⏳ **역전파 진행 중... ({layer_name} 도달)**\n
        현재 위치의 오차(기울기) 크기: **{current_g:.6f}**\n
        👉 **좌측 패널의 버튼을 눌러 다음 층으로 전파시키세요.**
        """)
    else:
        final_grad = grads[1] # H1 gradient
        if activation == 'Sigmoid':
            st.error(f"""
            🚨 **기울기 소멸(Vanishing Gradient) 발생!**
            
            - 출력층 초기 오차: **1.0**
            - 첫 번째 은닉층(H1)에 도달한 최종 오차: **{final_grad:.8f}**
            
            **[결과 분석]**
            오차 신호가 층을 지날 때마다 연쇄법칙에 의해 **최대 0.25배씩 계속 곱해진 결과**, 앞쪽 층에는 신호가 거의 0으로 소멸해 버렸습니다. 
            이렇게 되면 앞쪽 층의 가중치 파이프들은 무엇을 어떻게 고쳐야 할지(기울기) 알 수 없게 되어, 사실상 **학습이 멈춘 뇌사 상태**가 됩니다. 
            """)
        else:
            st.success(f"""
            🎉 **기울기 보존 성공 (ReLU의 힘)!**
            
            - 출력층 초기 오차: **1.0**
            - 첫 번째 은닉층(H1)에 도달한 최종 오차: **{final_grad:.1f}**
            
            **[결과 분석]**
            ReLU는 양수 영역에서 항상 **1.0의 기울기**를 반환하므로, 층이 10개, 100개로 깊어져도 1.0이 계속 곱해져 오차가 전혀 줄어들지 않고 온전히 앞쪽까지 도달했습니다!
            이 덕분에 아주 앞쪽에 있는 은닉층 가중치들도 자신의 잘못을 명확히 깨닫고 팍팍 수정될 수 있습니다. 
            이 단순한 원리 덕분에 오늘날 우리가 아는 거대한 딥러닝 모델의 학습이 비로소 가능해졌습니다.
            """)
            st.balloons()
