import streamlit as st



st.title("딥러닝 핵심 개념 정리")
st.markdown("---")

st.header("1. 퍼셉트론 (Perceptron)")
st.subheader("수학적 퍼셉트론")
st.markdown("""
* **가중치 (w, Weight):** 입력의 중요도를 나타냅니다. ("얼마나 중요한가")
* **편향 (b, Bias):** 활성화 기준점입니다. ("얼마나 까다로운가")
* **활성화 함수 $\sigma(z)$:** Sigmoid 함수 등을 사용하여 "(0, 1) 확률로 변환"합니다.

이를 통해 **AND 와 OR 게이트를 예측 가능**합니다.
결과적으로, **가중치와 편향의 조합 = 학습된 지식**이라고 할 수 있습니다.
""")

st.markdown("---")

st.header("2. AND/OR/XOR 퍼셉트론과 단층 퍼셉트론의 한계")
st.subheader("XOR 단층 퍼셉트론의 한계")
st.markdown("""
* 가중치와 편향을 어떻게 바꾸더라도 예측이 안됩니다.
* 그 이유는 **선형 분리가 불가능**하기 때문입니다.
* **해결 방법:** 층을 하나 더 쌓으면 **비선형 경계**로 나눌 수 있습니다.
""")

st.markdown("---")

st.header("3. MLP (다층 퍼셉트론, Multi-Layer Perceptron)")
st.markdown("""
층이 깊어질수록 **추상적 표현이 가능**해집니다.
* *예: 1층 -> 엣지, 방향선 / 2층 -> 눈, 코, 입 / 3층 -> 얼굴 전체*
* "우리가 눈을 찾아라 알려주지 않아도, 신경망이 스스로 발견"합니다.

**층 구조**
* **입력층:** 데이터 전달 (연산 X)
* **은닉층:** 중간 표현 (Feature) 자동 학습
* **출력층:** 최종 예측 (클래스 확률)
""")

st.subheader("왜 활성화 함수가 반드시 필요한가?")
st.markdown("""
* 활성화 함수 없이 Layer를 쌓으면, 수학적으로 선형 모델 1개와 동일합니다.
* 즉, **비선형 활성화 함수가 있어야 층이 의미를 가집니다.**

**활성화 함수 종류**
* Sigmoid
* Softmax
* ReLU
""")

st.subheader("Softmax")
st.markdown("""
**왜 exp(지수 함수)를 사용하는가?**
1. **항상 양수:** 확률의 조건 충족
2. **차이 증폭:** 가장 큰 z 값이 두드러짐
3. **미분 가능:** 역전파(Backpropagation) 가능
""")

st.markdown("---")

st.header("4. TensorFlow / Keras")
st.subheader("4-1. Tensor란?")
st.markdown("""
* **딥러닝의 데이터 단위**입니다.
* Numpy는 CPU 연산을 하지만, **TF Tensor는 GPU 연산이 가능**하고 **자동 미분(역전파)으로 편리**합니다.
""")

st.subheader("4-2. Keras 5단계 패턴")
st.markdown("""
고수준 API로 마치 레고 블록을 조립하듯 사용합니다.

1. **`Sequential` 정의** (모델 뼈대 생성)
2. **`compile`** (학습 방법 설정: 손실 함수, 옵티마이저 등)
3. **`fit`** (데이터 학습)
4. **`evaluate`** (모델 평가)
5. **`summary`** (모델 구조 확인)
""")

st.markdown("---")

st.header("5. ML vs DL (sklearn digits 예시)")
st.markdown("""
**ML vs DL 비교**
* **ML:** Logistic Regression
* **DL:** Keras Dense Layer

**수학적으로 완전히 같은 구조!**
* LogReg: $P(y =k \mid x) = \\text{softmax}(Wx+b)_k$
* Dense(10, softmax): $a = \\text{softmax}(Wx+b)$

결론적으로, **Dense 단층 = 다중 클래스 로지스틱 회귀**입니다.
""")

st.markdown("---")

st.header("6. Flatten")
st.markdown("""
**왜 Flatten이 필요한가?**
* `Dense` 레이어는 **1D 입력만 받습니다.**
* 따라서 2D 이미지 데이터를 다룰 때는 반드시 **Flatten 해야 합니다.** (`.reshape`와 유사한 역할)
* *(참고: CNN의 경우는 `Conv2D`를 통해 2D 형태 그대로 입력을 받을 수 있습니다.)*
""")

st.markdown("---")

st.header("💡 오늘의 큰 질문 요약")
st.info("""
**1. 퍼셉트론?**
* 가중치·편향 조합 = 학습된 지식

**2. 왜 층을 쌓아야 하는가?**
* 단층: 직선 경계만 가능 (XOR 해결 불가)
* 다층: 비선형 경계 가능
* **주의:** 활성화 함수가 없으면 층을 쌓는 의미가 없음

**3. Keras로 어떻게 만드는가? (5단계)**
1. `Sequential` 정의
2. `compile`
3. `fit`
4. `evaluate`
5. `summary`
""")


st.info("""입력값 784 Dense 256 이면 파라미터는 몇일까?

해설 : 파라미터 = 입력값 * dense + 256

입력 노드 784개와 Dense 층의 출력 노드 256개를 연결하는 레이어의 총 파라미터 수는 200,960개입니다.이 파라미터 수는 가중치(Weights)와 편향(Biases)의 합으로 계산됩니다.가중치 (Weights): \(784 \times 256 = 200,704\)편향 (Biases): 출력 노드마다 1개씩 총 \(256\)개총합: \(200,704 + 256 = 200,960\)이 수식은 아래의 일반적인 Dense 층 파라미터 계산 공식을 따릅니다.\((입력 노드 수 \times 출력 노드 수) + 출력 노드 수\)

파라미터가 올라간다고 정확도가 꼭 올라가는 것은 아님

""")

