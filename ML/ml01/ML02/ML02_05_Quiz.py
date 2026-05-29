import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="회귀 알고리즘 퀴즈",
    page_icon="📝",
    layout="wide"
)

st.title("🧠 ML02 개념 점검 퀴즈")
st.markdown("**데이터 전처리, k-NN 및 선형 회귀, 경사 하강법, 다항 회귀, 규제 모델(Ridge/Lasso), 하이퍼파라미터**의 핵심 개념 퀴즈")
st.divider()

# 퀴즈 목록 정의
quizzes = [
    {
        "id": 1,
        "question": "Q1. StandardScaler 전처리 기법의 필요성을 '단위 스케일 불균형'과 '데이터 누수(Data Leakage)' 관점에서 바르게 설명하지 못한 것은?",
        "options": [
            "단위 스케일 불균형 상태에서 거리 기반 모델을 적용하면, 수치 범위가 큰 피처가 거리 계산을 주도하여 다른 피처의 영향력을 완전히 가려버린다.",
            "데이터 누수를 방지하기 위해 전체 데이터셋(훈련+테스트)에 대해 fit_transform을 단 한 번만 일괄 적용하여 동일한 통계량으로 표준화하는 것이 가장 올바른 학습 방법이다.",
            "테스트 데이터셋을 변환할 때는 반드시 훈련 데이터셋의 평균과 표준편차를 기준으로 `.transform()`만 적용하여 평가용 데이터 정보가 유출되지 않도록 조치해야 한다.",
            "StandardScaler를 적용하면 각 피처의 평균은 0, 표준편차는 1이 되도록 변환하여 데이터의 고유 분포 편차로 인한 모델 왜곡을 사전에 예방한다."
        ],
        "answer": 1,
        "explanation": "전체 데이터셋에 대해 fit_transform을 한 번에 일괄 적용하면 테스트 데이터의 정보(평균 및 표준편차)가 스케일러에 반영되는 **데이터 누수(Data Leakage)**가 발생합니다. 이는 미래의 시험지 정보를 미리 훔쳐보는 것과 같으므로 올바르지 않습니다. 반드시 **훈련 데이터셋으로만 fit_transform을 수행**하고, **테스트 데이터셋은 transform만** 적용해야 합니다."
    },
    {
        "id": 2,
        "question": "Q2. k-NN 알고리즘을 활용한 '분류(Classification)'와 '회귀(Regression)'의 핵심적인 작동 차이 및 결정 계수($R^2$)의 의미를 바르게 설명한 것은?",
        "options": [
            "분류는 연속적인 수치를 예측하고 회귀는 이웃의 다수결에 의해 최종 범주를 결정하며, $R^2$가 0에 가까울수록 모델의 예측 성능이 완벽함을 뜻한다.",
            "회귀는 가장 가까운 이웃 $k$개 타깃값들의 평균을 구해 연속적인 실수 수치로 예측하며, $R^2$가 1.0에 도달하면 정답을 완벽하게 맞추었음을 의미한다.",
            "$R^2$ 점수가 음수(예: -144)로 나오는 것은 코딩 상의 치명적인 구문 에러가 발생했음을 뜻하므로 즉시 소스코드를 전체 재설행하여 에러를 디버깅해야 한다.",
            "k-NN 회귀는 학습된 직선 방정식을 추종하기 때문에 훈련 데이터 범위를 벗어나는 새로운 외부 영역에 대해서도 외삽(Extrapolation) 예측을 완벽하게 수행해 낼 수 있다."
        ],
        "answer": 1,
        "explanation": "회귀(Regression)는 이웃 $k$개 타깃값들의 평균을 도출해 연속적인 실수 수치값을 예측합니다. R²(결정 계수)는 모델의 상대적인 예측 성능 평가 지표로 1.0에 가까울수록 오차가 없는 완벽한 예측을 의미하며, 0.0은 항상 타깃의 평균값으로만 예측하는 기초 모델 수준임을, 음수(<0)는 평균 예측보다도 성능이 처참한 상태(기본적인 흐름조차 맞추지 못함)임을 나타냅니다. (k-NN 회귀는 방정식 학습이 아니므로 외삽이 불가능합니다.)"
    },
    {
        "id": 3,
        "question": "Q3. 선형 회귀(Linear Regression)에서 찾고자 하는 '최적 직선'의 수학적 의미와 손실 함수인 '평균 제곱 오차(MSE)'의 타당성으로 가장 적절한 것은?",
        "options": [
            "최적 직선은 데이터 포인트들의 Y절편값을 전부 0으로 강제하는 직선이며, MSE는 오차 부호의 제곱을 피하기 위해 단순 절대값만 합산하는 지표이다.",
            "최적 직선은 실제 값과 예측값 간 오차의 제곱합(MSE)을 최소화하는 기울기 $w$와 절편 $b$를 가진 방정식이며, MSE 계산 시 오차를 제곱하는 것은 부호 상쇄를 막고 큰 오차에 강력한 패널티를 부과하기 위함이다.",
            "MSE 손실 함수의 오차를 제곱하는 유일한 과학적 이유는 기울기가 음수일 때 수학적 계산 오류(런타임 에러)가 발생하는 것을 막기 위해서다.",
            "선형 회귀 모델은 훈련 데이터 범위를 벗어난 새로운 영역(외삽)에 대해서도 k-NN 모델처럼 안정적인 고정 평균 최댓값만을 예측값으로 고정 반환한다."
        ],
        "answer": 1,
        "explanation": "선형 회귀의 목표는 실제 데이터와 예측선 사이의 수직 오차의 제곱합인 MSE(Mean Squared Error)를 최소화하는 파라미터($w$, $b$)를 탐색하는 것입니다. 오차를 제곱하는 과학적 타당성은: 1) 양수와 음수 오차가 상쇄되는 현상을 방지하고, 2) 큰 오차에 대해 더 강력한 제곱 형태의 패널티를 부여하여 예측값이 실제 데이터 경향에서 크게 이탈하는 것을 막기 위함입니다."
    },
    {
        "id": 4,
        "question": "Q4. 손실 함수를 점진적으로 최소화하는 '경사 하강법(Gradient Descent)'의 파라미터 업데이트 원리와 학습률($\\alpha$)에 따른 모델 반응으로 올바른 것은?",
        "options": [
            "가중치 $w$는 기울기의 반대 방향($w \\leftarrow w - \\alpha \\frac{\\partial J}{\\partial w}$)으로 업데이트되며, 학습률 $\\alpha$가 지나치게 크면 최적점에 수렴하지 못하고 골짜기 위로 발산(Overshooting)하게 된다.",
            "경사 하강법은 항상 정규 방정식처럼 단 한 번의 행렬 연산만으로 최적해를 도출하므로 학습률 $\\alpha$를 튜닝할 필요가 전혀 없다.",
            "학습률 $\\alpha$가 극도로 작으면 한 번에 업데이트 보폭이 매우 크게 도약하므로 아주 빠른 속도로 최저 지점에 수렴할 수 있다.",
            "가중치는 항상 손실 함수의 기울기 자체의 방향($w \\leftarrow w + \\alpha \\frac{\\partial J}{\\partial w}$)과 동일하게 더해 나감으로써 전체 오차를 지속적으로 극대화한다."
        ],
        "answer": 0,
        "explanation": "경사 하강법은 현재 가중치 위치에서 손실 함수 $J$를 미분한 기울기(Gradient)의 반대 방향으로 조금씩 업데이트해 손실 최하점으로 수렴해 나갑니다 ($w \\leftarrow w - \\alpha \\frac{\\partial J}{\\partial w}$). 이때 보폭 역할을 하는 학습률 $\\alpha$가 너무 크면 최적점을 지나쳐 반대편으로 튀어 올라 발산(Overshooting)하며, 너무 작으면 업데이트 속도가 현저히 느려져 수렴에 실패할 수 있습니다."
    },
    {
        "id": 5,
        "question": "Q5. 다항 회귀(Polynomial Regression) 모델을 학습할 때, '과적합(Overfitting)'이 발생하는 핵심적인 유도 조건과 증상으로 가장 알맞은 것은?",
        "options": [
            "다항식의 차수(Degree)를 극도로 낮추어 피처의 수를 지나치게 단순화할 때, 테스트 세트 점수와 훈련 세트 점수가 모두 0.1 이하로 추락하는 모델 비활성화 현상",
            "피처의 차수(Degree)를 지나치게 확장하여 피처 수가 데이터 수보다 팽창할 때, 모델이 훈련 데이터의 자잘한 노이즈와 특이 오차까지 완벽히 암기(Train $R^2 \\approx 1.0$)하여 정작 처음 보는 테스트 데이터에는 처참한 예측 성능(Test $R^2 \\ll 0$)을 보이는 현상",
            "데이터 전처리 과정에서 StandardScaler를 철저히 fit-transform 분리하여 사용함으로써 데이터의 정규성 분포가 비정상적으로 강화되는 현상",
            "모델이 단순하여 훈련 데이터의 근본적인 선형 패턴조차 전혀 대변해 내지 못하고 훈련 세트 $R^2$가 0.5 미만에 머물러 있는 과소적합 상태"
        ],
        "answer": 1,
        "explanation": "모델의 유연성(복잡도)이 데이터 분포보다 과도하게 높을 때(예: 다항 회귀의 degree를 너무 높여 특성 수가 과도하게 팽창할 때) 과적합(Overfitting)이 일어납니다. 모델은 일반화된 패턴 대신 훈련 데이터에 완벽히 피팅되도록 미세한 노이즈(Noise)까지 암기하게 되어, 훈련 세트 $R^2$ 점수는 1.0에 육박하지만 테스트 세트 R² 점수는 심각한 음수로 폭락하게 됩니다."
    },
    {
        "id": 6,
        "question": "Q6. 특성 과적합을 제어하기 위해 적용하는 규제 모델인 'Ridge(L2 규제)'와 'Lasso(L1 규제)'의 규제 원리와 전처리 필요성에 대한 설명으로 올바르지 않은 것은?",
        "options": [
            "Ridge 회귀는 계수의 제곱 합에 대한 패널티를 더해 가중치의 절대적 크기를 부드럽게 억제하지만, 모든 피처를 소량이라도 끝까지 활용한다.",
            "Lasso 회귀는 계수의 절댓값 합에 패널티를 부과하며, 유의미하지 않거나 불필요한 피처의 가중치를 정확하게 0으로 만들어 자동 특성 선택(Sparsity)을 수행한다.",
            "규제 강도 매개변수 $\\alpha$가 커질수록 모델에 강력한 패널티가 가해져 모델 가중치들이 전체적으로 크게 억제되어 단순해지며, 극단적으로 커지면 과소적합 상태가 유도된다.",
            "Ridge와 Lasso는 회귀 계수(가중치)의 크기를 직접 규제하므로, 각 특성의 단위 스케일 차이가 크더라도 별도의 StandardScaler 표준화 처리를 할 필요가 전혀 없다."
        ],
        "answer": 3,
        "explanation": "Ridge와 Lasso는 회귀 계수(가중치) 자체의 수치 크기를 직접 억제하여 패널티를 줍니다. 따라서 특성들끼리 스케일(단위 범위)이 다르면 가중치 크기도 스케일에 비례하여 왜곡되므로, 특정 피처에 편향된 규제가 걸리는 부작용이 있습니다. 그러므로 **Ridge와 Lasso 규제 모델 적용 전에는 반드시 StandardScaler를 거쳐 스케일을 동일 정규분포 척도로 통일**시켜 주어야 합니다."
    },
    {
        "id": 7,
        "question": "Q7. 머신러닝의 내부 관리 변수인 '파라미터(Parameter)'와 사용자가 지정해야 하는 '하이퍼파라미터(Hyperparameter)'의 차이점으로 가장 올바르지 않은 것은?",
        "options": [
            "파라미터는 모델이 훈련 데이터로부터 학습 과정(.fit())을 거쳐 스스로 탐색하고 갱신하여 획득하는 가중치($w$, coef_)와 편향($b$, intercept_) 등의 값이다.",
            "하이퍼파라미터는 모델의 최적 동작 및 복잡도를 제어하기 위해 학습을 시작하기 전 사람이 직접 설정해야 하는 값이다.",
            "k-NN의 이웃 수 $k$, 다항 회귀의 차수 degree, 규제 강도 $\\alpha$ 등은 분석가가 직접 값을 지정하고 비교 탐색해야 하는 하이퍼파라미터다.",
            "하이퍼파라미터 또한 알고리즘 내부에서 `.fit()` 호출 시 훈련 데이터에 맞춰 자동으로 글로벌 최적 상수가 수학적으로 자동 결정되므로 사람이 전혀 개입할 필요가 없다."
        ],
        "answer": 3,
        "explanation": "하이퍼파라미터(예: k, degree, alpha)는 사람이 머신러닝 모델의 복잡도 제어 및 일반화 최적화를 위해 직접 대입하고 탐색해가며 튜닝해야 하는 설정값입니다. 모델 스스로 학습 과정에서 결정해낼 수 없으므로, 사람이 다양한 탐색(그리드 서치 등)이나 시각화를 통해 적정 최적치를 실험적으로 규명하고 지정해야 마땅합니다."
    }
]

# 각 문제별 제출 상태 및 선택한 답변을 세션 스테이트에서 개별 관리합니다.
for idx, q in enumerate(quizzes):
    q_id = q["id"]
    submitted_key = f"q2_{q_id}_submitted"
    choice_key = f"q2_{q_id}_choice"
    
    # 세션 스테이트 변수 초기화
    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False
    if choice_key not in st.session_state:
        st.session_state[choice_key] = 0
        
    st.markdown(f"### {q['question']}")
    
    if not st.session_state[submitted_key]:
        # [제출 전 상태] 라디오 버튼과 개별 제출 버튼 표시
        user_choice = st.radio(
            f"선택지 (Q{q_id})",
            options=q["options"],
            index=st.session_state[choice_key],
            key=f"radio2_{q_id}",
            label_visibility="collapsed"
        )
        # 사용자가 클릭한 옵션의 인덱스를 실시간 저장
        st.session_state[choice_key] = q["options"].index(user_choice)
        
        # 개별 제출 버튼
        if st.button(f"📝 Q{q_id} 정답 제출", key=f"btn2_{q_id}"):
            st.session_state[submitted_key] = True
            st.rerun()
            
    else:
        # [제출 완료 상태] 정오 판정 결과, 해설 상자, 개별 다시 풀기 버튼 표시
        chosen_idx = st.session_state[choice_key]
        user_choice_text = q["options"][chosen_idx]
        correct_choice_text = q["options"][q["answer"]]
        is_correct = chosen_idx == q["answer"]
        
        # 1. 정오 확인 표시
        if is_correct:
            st.success(f"🟢 **정답입니다!**\n\n- **선택한 답**: {user_choice_text}")
        else:
            st.error(f"🔴 **오답입니다.**\n\n- **선택한 답**: {user_choice_text}\n- **올바른 정답**: {correct_choice_text}")
            
        # 2. 상세 해설 상자
        st.info(f"💡 **상세 해설:**\n\n{q['explanation']}")
        
        # 3. 개별 다시 풀기 버튼
        if st.button(f"🔄 Q{q_id} 다시 풀기", key=f"retry2_{q_id}"):
            st.session_state[submitted_key] = False
            st.session_state[choice_key] = 0
            st.rerun()
            
    st.markdown("---")
