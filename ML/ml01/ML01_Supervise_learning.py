'''
지식 목표

AI·ML·DL의 계층 관계를 설명하고 세 가지 학습 유형(지도·비지도·강화)을 예시와 함께 구분할 수 있다. 
지도학습 파이프라인(데이터→특성→분리→학습→예측→평가)의 각 단계를 손으로 그리고 설명할 수 있다. 
train_test_split이 왜 필요한지 일반화(Generalization) 개념과 연결해 설명할 수 있다. 
Feature와 Label의 차이를 실습 데이터에 1:1로 매핑해 설명할 수 있다
'''

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# Matplotlib 한글 폰트 설정 (Windows 기준: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 도미와 빙어 데이터 정의
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7,
                31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5,
                34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0,
                38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0,
                500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0,
                610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0,
                714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2,
                12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2,
                13.4, 12.2, 19.7, 19.9]



st.write("## " + "학습 목표")
st.write("1. AI·ML·DL의 계층 관계를 설명하고 세 가지 학습 유형(지도·비지도·강화)을 예시와 함께 구분할 수 있다.")
st.write("2. 지도학습 파이프라인(데이터→특성→분리→학습→예측→평가)의 각 단계를 설명할 수 있다.") 
st.write("3. train_test_split이 왜 필요한지 일반화(Generalization) 개념과 연결해 설명할 수 있다.")
st.write("4. Feature와 Label의 차이를 실습 데이터에 1:1로 매핑해 설명할 수 있다.")

st.divider()

st.write("## " + "1. AI, ML, DL")
st.write("1. AI : 인간처럼 생각하려는 기계를 만드려는 모든 시도")
st.write("2. ML : 데이터에서 스스로 패턴을 학습하는 AI")
st.write("3. DL : 뇌의 신경망을 모방한 ML - 이미지, 음성, 언어")

st.write("#### " + "ML은 지도, 비지도, 강화 학습으로 구분된다.")
image_path = os.path.join(os.path.dirname(__file__), "data", "type_of_machinelearning.PNG")
st.image(image_path, caption='type of ML')

st.write("#### " + "지도 학습")
st.write("지도 학습은 레이블(정답)이 있는 데이터를 보고 학습하는 방식")

st.write("#### " + "비지도 학습")
st.write("비지도 학습은 레이블(정답)이 없는 데이터를 보고 학습하는 방식")

st.write("#### " + "강화 학습")
st.write("강화 학습은 보상을 통해 학습하는 방식")

st.divider()


st.write("## " + "2. 지도 학습 파이프라인")
image_path = os.path.join(os.path.dirname(__file__), "data", "pipeline of supervised learning.PNG")
st.image(image_path, caption='pipeline of supervised learning')
st.write("데이터 수집 : 레이블이 있는 데이터 수집")
st.write("특성 추출 : 레이블 예측에 도움이 되는 특성 추출")
st.write("데이터 분할 : 훈련 데이터와 테스트 데이터로 분할")
st.write("모델 학습 : 훈련 데이터로 모델 학습")
st.write("모델 예측 : 테스트 데이터로 모델 예측")
st.write("모델 평가 : 테스트 데이터로 모델 성능 평가")
st.write("모델 개선 : 모델 성능 개선")

st.divider()

st.write("## " + "3. 왜 train_test_split이 필요한가?")

st.write("### 일반화 (Generalization)")
st.markdown("""
머신러닝의 궁극적인 목표는 우리가 가진 데이터만 완벽하게 맞추는 것이 아니라, 
**학습할 때 보지 못한 완전히 새로운 데이터 (Unseen Data)** 가 들어왔을 때도 정확하게 예측하는 것입니다. 
이를 **일반화 (Generalization)** 라고 부르며, 머신러닝 모델의 실제 성능을 결정하는 가장 중요한 지표입니다.
""")

st.write("### 과대적합 (Overfitting)")
st.markdown("""
만약 우리가 수집한 도미와 빙어 데이터를 모두 학습에 사용하고, 똑같은 데이터로 모델의 정확도를 평가한다면
- 모델은 새로운 규칙을 배우는 대신, 단순히 데이터 자체를 통째로 암기(Overfitting)해 버립니다.
- 이미 외운 시험지를 바탕으로 시험을 치기 때문에 평가 정확도는 100%가 나오지만,
- 실제 현장에서 새로운 물고기 데이터가 들어왔을 때는 엉터리 예측을 하게 된다.

이를 방지하기 위해 우리는 전체 데이터를 학습용(Train Set)과 평가용(Test Set)으로 엄격히 분할해야 합니다.
""")

st.write("### 도미 & 빙어 데이터를 활용한 `train_test_split` 실습")

# 1. 도미와 빙어 데이터를 하나의 데이터셋으로 합치기
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight

# Feature (특성): 모델이 학습할 데이터 (2차원 형태)
X = [[l, w] for l, w in zip(fish_length, fish_weight)]
# Label (정답/타깃): 도미=1, 빙어=0
y = [1] * len(bream_length) + [0] * len(smelt_length)

# 2. train_test_split으로 분할 (훈련:테스트 = 75:25 비율)
# stratify=y 를 설정해 도미와 빙어의 비율이 훈련/테스트에 골고루 섞이게 만듭니다.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 결과를 Streamlit에 깔끔하게 출력
col1, col2 = st.columns(2)
with col1:
    st.metric(label="전체 물고기 수", value=f"{len(X)} 마리", delta="도미 35 + 빙어 14")
with col2:
    st.metric(label="훈련 세트 (Train Set)", value=f"{len(X_train)} 마리 (75%)")
    st.metric(label="테스트 세트 (Test Set)", value=f"{len(X_test)} 마리 (25%)")

# 분할된 데이터를 인터랙티브 데이터프레임으로 표현
st.write("#### 📑 분할된 테스트 세트 (Test Set) 확인하기 (총 13마리)")
st.markdown("테스트 세트는 모델 학습이 **완전히 끝난 후, 일반화 능력을 검증하기 전까지는 절대 들여다보지 않는 영역**입니다.")

test_df = pd.DataFrame(X_test, columns=['길이(cm)', '무게(g)'])
test_df['종류(Label)'] = ['도미 (1)' if val == 1 else '빙어 (0)' for val in y_test]
st.dataframe(test_df.style.background_gradient(cmap='Blues', subset=['길이(cm)', '무게(g)']), use_container_width=True)


st.divider()



st.write("## " + "4. Feature와 Label의 차이")

st.markdown("""
머신러닝(특히 지도학습) 모델을 설계할 때는 데이터를 **Feature (특성)** 와 **Label (정답/타깃)** 로 명확히 구분해야 합니다.
이를 도미와 빙어 실습 데이터에 1:1로 매핑하여 정리하면 다음과 같습니다.
""")

# 1:1 매핑 표 구성
st.markdown("""
| 개념 | 머신러닝의 역할 (의미) | 물고기 데이터에 비유 | 수학적 표기 및 형태 |
| :--- | :--- | :--- | :--- |
| **Feature (특성)** | 모델이 학습하고 예측할 때 단서가 되는 **입력값** | 물고기의 **길이 (length)** 와 **무게 (weight)** | $X$ (보통 2차원 데이터) |
| **Label (정답/타깃)** | 모델이 최종적으로 맞추어야 할 **목표 정답** | 물고기의 종류인 **도미 (1)** 또는 **빙어 (0)** | $y$ (보통 1차원 데이터) |
""")

st.write("### 🔍 실제 데이터 1:1 매핑 예시 (한 마리 기준)")
st.markdown("""
우리가 모은 데이터셋의 첫 번째 물고기를 꺼내서 분석해 봅시다.

```python
# 1번째 물고기 데이터
length = 25.4  # 길이
weight = 242.0 # 무게
target = 1     # 도미(1)
```

이 한 마리의 물고기 데이터를 머신러닝 관점으로 매핑하면 다음과 같습니다.
- **Feature** ($X$): `[25.4, 242.0]` (길이와 무게라는 2가지 힌트)
- **Label** ($y$): `1` (이 물고기는 '도미'라는 정답)

> **💡 핵심 요약**
> - **Feature** 는 문제를 풀기 위해 주어지는 **힌트** 이고,
> - **Label** 은 모델이 최종적으로 맞춰야 할 **문제의 정답** 입니다.
""")





st.divider()
# 도미와 빙어 데이터
st.write("### 도미와 빙어 전체 데이터 시각화")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(bream_length, bream_weight, label='도미(Bream)', color='#2F80ED', alpha=0.8, edgecolors='w', s=80)
ax.scatter(smelt_length, smelt_weight, label='빙어(Smelt)', color='#F2994A', alpha=0.8, edgecolors='w', s=80)
ax.set_title('도미와 빙어의 길이 및 무게 분포', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('길이(cm)', fontsize=12)
ax.set_ylabel('무게(g)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

st.pyplot(fig)


