import streamlit as st

st.title("📝 ML 03: 분류 알고리즘 및 평가지표 퀴즈")
st.markdown("분류 모델, 혼동 행렬(Confusion Matrix), 불균형 데이터 평가지표, 그리고 Bias-Variance 진단에 대한 핵심 개념을 복습하는 퀴즈입니다.")

# 퀴즈 데이터
quizzes = [
    {
        "question": "Q1. 로지스틱 회귀(Logistic Regression) 모델이 이진 분류와 다중 분류를 수행할 때 내부적으로 각각 어떤 함수를 사용하여 확률(0~1)을 계산할까요?",
        "options": [
            "(A) Sigmoid / Softmax",
            "(B) ReLU / Softmax",
            "(C) Tanh / Sigmoid",
            "(D) Softmax / Sigmoid"
        ],
        "answer": "(A) Sigmoid / Softmax",
        "explanation": "정답은 (A)입니다! 이진 분류에서는 `Sigmoid` 함수를 사용해 z값을 0~1 사이의 확률로 압축하고, 다중 분류에서는 `Softmax` 함수를 사용해 모든 클래스 확률의 합이 딱 1(100%)이 되도록 배분합니다."
    },
    {
        "question": "Q2. 병원에서 암 진단 모델을 개발했습니다. '실제로는 암 환자인데, 정상이라고 오진해서 환자를 집에 돌려보낸 치명적인 경우'는 혼동 행렬(Confusion Matrix)의 어떤 셀에 해당하며, 제 몇 종 오류일까요?",
        "options": [
            "(A) False Positive (FP), 제 1종 오류",
            "(B) False Negative (FN), 제 1종 오류",
            "(C) False Negative (FN), 제 2종 오류",
            "(D) True Negative (TN), 제 2종 오류"
        ],
        "answer": "(C) False Negative (FN), 제 2종 오류",
        "explanation": "정답은 (C)입니다! 실제 암(Positive)인데 정상(Negative)으로 잘못(False) 예측했으므로 **False Negative(FN)**입니다. 이를 통계학에서는 **제 2종 오류**라고 부르며, 암 진단 등 생명이 직결된 의료 도메인에서는 무조건 최소화해야 하는 가장 위험한 오류입니다."
    },
    {
        "question": "Q3. 사망자가 99%, 생존자가 1%인 매우 불균형한 재난 데이터가 있습니다. 모델 학습 없이 무조건 '사망(0)'으로만 찍는 바보 모델을 만들었을 때, 다음 중 틀린 설명은 무엇일까요?",
        "options": [
            "(A) 이 모델의 정확도(Accuracy)는 99%로 매우 높게 나옵니다.",
            "(B) 생존자를 찾는 것이 목적일 때, 이 모델의 재현율(Recall)은 0% 입니다.",
            "(C) 이 모델의 ROC-AUC 스코어는 1.0(완벽)에 가까울 것입니다.",
            "(D) 불균형 데이터에서는 정확도만 보면 모델 성능을 착각하기 쉽습니다."
        ],
        "answer": "(C) 이 모델의 ROC-AUC 스코어는 1.0(완벽)에 가까울 것입니다.",
        "explanation": "정답은 (C)입니다! ROC-AUC는 진짜 양성 비율(TPR)과 거짓 양성 비율(FPR)의 변화를 고려하여 임계값에 상관없는 전반적인 분류 성능(랭킹)을 평가합니다. 한쪽으로만 무조건 찍는 모델은 예측 확률 분별력이 없으므로 AUC가 0.5(동전 던지기 수준)가 됩니다. 따라서 AUC 1.0 이라는 설명은 틀렸습니다."
    },
    {
        "question": "Q4. 모델을 학습시킨 결과, Train Score는 98%가 나왔지만 Test Score는 62%로 뚝 떨어졌습니다. 현재 이 모델의 상태는 무엇이며 어떤 조치가 필요할까요?",
        "options": [
            "(A) High Bias (과소적합) - 모델을 더 단순화해야 한다.",
            "(B) High Variance (과적합) - 규제(Regularization)를 강화하거나 데이터를 더 모아야 한다.",
            "(C) 적절한 피팅 - 즉시 서비스에 배포해도 좋다.",
            "(D) Data Leakage (데이터 누수) - 테스트 데이터를 학습 단계에 포함시켜야 한다."
        ],
        "answer": "(B) High Variance (과적합) - 규제(Regularization)를 강화하거나 데이터를 더 모아야 한다.",
        "explanation": "정답은 (B)입니다! 훈련 세트에는 엄청나게 잘 맞지만(Train 98%), 처음 보는 새로운 데이터에는 죽을 쑤는(Test 62%) 상태를 **과대적합(High Variance)**이라고 합니다. 이때는 모델의 복잡도를 낮추는 규제(L1, L2 등)를 가하거나 훈련 데이터를 더 많이 수집해야 합니다."
    },
    {
        "question": "Q5. 점진적 학습(SGDClassifier 등)에서 에포크(Epoch)를 늘려가며 학습할 때, '조기 종료(Early Stopping)'를 해야 하는 가장 최적의 시점은 언제일까요?",
        "options": [
            "(A) Train Score가 최고점에 도달했을 때",
            "(B) Train Score와 Test Score가 똑같아졌을 때",
            "(C) Test Score가 최고점에 도달하고 하락하기 시작하는(과적합이 시작되는) 직전의 시점",
            "(D) 모델 학습 시간이 개발자가 정해둔 1시간을 넘겼을 때"
        ],
        "answer": "(C) Test Score가 최고점에 도달하고 하락하기 시작하는(과적합이 시작되는) 직전의 시점",
        "explanation": "정답은 (C)입니다! 에포크가 늘어날수록 Train 점수는 계속 오르지만, 어느 순간부터는 훈련 데이터에만 과도하게 맞춰지면서 Test 점수가 오히려 꺾여 내려가기 시작합니다(과적합 발생). 따라서 **Test Score(검증 점수)가 최고점을 찍는 그 순간** 학습을 딱 멈추는 것이 가장 좋습니다."
    },
    {
        "question": "Q6. [추가] 유방암 진단 모델에서 의사가 임계값(Threshold) 커트라인을 0.5(50%)에서 0.3(30%)으로 대폭 낮췄습니다. 이때 발생하는 트레이드오프(Trade-off) 현상으로 올바른 것은?",
        "options": [
            "(A) 모델의 정확도(Accuracy)가 무조건 100%로 상승한다.",
            "(B) 정상 환자를 암으로 오진하는 일(FP)이 줄어들고, 정밀도(Precision)가 높아진다.",
            "(C) 진짜 암 환자를 놓치는 일(FN)이 줄어들고, 재현율(Recall)이 높아진다.",
            "(D) 재현율(Recall)과 정밀도(Precision)가 동시에 사이좋게 높아진다."
        ],
        "answer": "(C) 진짜 암 환자를 놓치는 일(FN)이 줄어들고, 재현율(Recall)이 높아진다.",
        "explanation": "정답은 (C)입니다! 커트라인을 0.3으로 관대하게 낮추면 \"조금만 의심스러워도 일단 암이라고 판정\"하게 됩니다. 덕분에 진짜 암 환자를 놓치는 치명적인 일(FN)이 크게 줄어들어 **Recall(재현율)은 올라갑니다**. (하지만 반대급부로 정상인인데 암이라고 오진 받는 사람(FP)이 늘어나 Precision(정밀도)은 떨어지는 Trade-off가 생깁니다.)"
    }
]

# 퀴즈 렌더링 로직
for i, q in enumerate(quizzes):
    st.subheader(f"{q['question']}")
    
    # 세션 상태 초기화
    state_key = f"quiz_{i}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None

    # 선택 라디오 버튼
    user_choice = st.radio(
        "정답을 선택하세요:",
        q["options"],
        key=f"radio_{i}",
        index=None
    )

    # 확인 버튼
    if st.button("정답 확인", key=f"btn_{i}"):
        if user_choice:
            st.session_state[state_key] = (user_choice == q["answer"])
        else:
            st.warning("보기 중 하나를 선택해 주세요!")

    # 결과 피드백 출력
    if st.session_state[state_key] is True:
        st.success(q["explanation"])
    elif st.session_state[state_key] is False:
        st.error(f"오답입니다! 다시 한번 생각해 보세요.")
    
    st.divider()

st.info("💡 **축하합니다!** 이 퀴즈들을 모두 이해하셨다면, 기계가 어떻게 분류를 하고 그것을 우리가 어떻게 제대로 평가해야 하는지에 대한 핵심적인 통찰력을 갖추신 겁니다!")
