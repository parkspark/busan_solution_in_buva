import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="머신러닝 퀴즈",
    page_icon="📝",
    layout="wide"
)

st.title("🧠 ML01 개념 점검 퀴즈")
st.markdown("**지도 학습, 일반화, train_test_split, Feature & Label**의 개념 퀴즈")
st.divider()

# 퀴즈 목록 정의
quizzes = [
    {
        "id": 1,
        "question": "Q1. 다음 중 지도학습(Supervised Learning)의 예시로 가장 올바른 것은?",
        "options": [
            "고객의 과거 구매 패턴을 분석하여 유사한 성향을 가진 고객들을 군집화하기",
            "이메일 텍스트와 스팸 여부 레이블을 활용해 새로운 이메일이 스팸인지 분류하기",
            "Netflix에서 정답 레이블 없이 사용자의 시청 이력 패턴만을 분석해 콘텐츠 추천하기",
            "대규모 뉴스 기사 데이터에서 인위적인 라벨링 없이 유사한 주제의 뉴스들끼리 묶기"
        ],
        "answer": 1,  # 0-indexed: 2번째 옵션이 정답
        "explanation": "지도학습은 예측할 정답(레이블)이 데이터에 포함된 경우입니다. 스팸 분류는 '스팸/정상'이라는 명확한 레이블(정답)이 있으므로 지도학습에 해당합니다. 반면 고객 군집화, 콘텐츠 추천, 뉴스 기사 묶기는 정답 레이블 없이 데이터 자체의 내재된 패턴을 발견하는 비지도학습(Unsupervised Learning)의 예시입니다."
    },
    {
        "id": 2,
        "question": "Q2. 아래 코드를 실행했을 때 kn.score(fish_data, fish_target)이 1.0이 나왔다. 이것은 완벽한 모델이라는 의미인가? 이유와 함께 답하시오.",
        "options": [
            "맞다 — 모든 데이터를 정확히 예측했으므로 완벽한 모델이다",
            "아니다 — 학습한 데이터와 평가한 데이터가 같아서 모델의 진짜 일반화 성능을 알 수 없다",
            "아니다 — scikit-learn의 score 함수에 치명적인 오류가 있어서 잘못 출력된 것이다",
            "상황에 따라 다르다 — 데이터의 전체 크기가 충분히 크다면 완벽한 모델로 볼 수 있다"
        ],
        "answer": 1,  # 0-indexed: 2번째 옵션이 정답
        "explanation": "kn.fit(fish_data, ...)으로 모델을 학습시킨 뒤 동일한 fish_data로 평가(score)를 받으면, 이미 정답을 외운 상태에서 시험을 치는 것과 같습니다. 이는 모델이 진짜 새로운 데이터(Unseen Data)에 잘 대응하는지 평가하는 '일반화 성능'을 알 수 없게 만듭니다. 반드시 학습용(Train)과 평가용(Test) 데이터를 나누어 평가해야 합니다."
    },
    {
        "id": 3,
        "question": "Q3. train_test_split을 사용할 때 random_state=42를 지정하는 이유는?",
        "options": [
            "42가 알고리즘의 예측 정확도를 높여주는 특별한 숫자이기 때문에",
            "코드를 재실행할 때마다 항상 동일한 방식으로 데이터가 분할되도록 만들기 위해",
            "테스트 데이터의 비율을 전체의 42%로 고정하여 분할하기 위해",
            "scikit-learn 라이브러리에서 반드시 입력해야 하는 필수 파라미터이기 때문에"
        ],
        "answer": 1,  # 0-indexed: 2번째 옵션이 정답
        "explanation": "random_state는 난수 생성을 위한 씨앗값(Seed)입니다. 이 값을 고정하면 코드를 다시 실행하거나 다른 컴퓨터에서 실행하더라도 항상 동일한 방식으로 무작위 분할을 수행합니다. 이를 통해 실험의 재현성(Reproducibility)을 보장할 수 있습니다. 42는 개발자들 사이에서 관행적으로 사용하는 대표적인 숫자일 뿐 특별한 기능은 없습니다."
    },
    {
        "id": 4,
        "question": "Q4. 다음 코드에서 잘못된 부분(A)을 찾고 그 이유를 가장 바르게 설명한 것은?\n\n```python\n# (A)\nkn.fit(fish_data, fish_target)\n# (B)\nkn.score(test_input, test_target)\n```",
        "options": [
            "(A)가 잘못됨 — 모델을 학습시킬 때는 전체 fish_data 대신 분리된 train_input과 train_target을 사용해야 한다",
            "(B)가 잘못됨 — 평가를 진행할 때는 모델이 학습했던 전체 fish_data와 fish_target을 넣어야 올바른 점수가 나온다",
            "(A)와 (B) 모두 잘못됨 — 데이터 분할 후에는 KNeighborsClassifier 대신 다른 모델을 사용해야만 한다",
            "문제없다 — 데이터를 분할한 뒤 전체 데이터로 학습하고 일부로 테스트하는 것이 올바른 머신러닝 파이프라인이다"
        ],
        "answer": 0,  # 0-indexed: 1번째 옵션이 정답
        "explanation": """`train_test_split`으로 분리한 의미가 있으려면 `.fit()`에 전체 `fish_data`를 쓰면 안 된다. 분리한 `train_input`, `train_target`만으로 학습해야 모델이 `test_input`을 "처음 보는 데이터"로 평가할 수 있다. 올바른 코드: `kn.fit(train_input, train_target)`. (B)의 score는 test 데이터로 평가하므로 올바르다. 데이터를 train_test_split을 통해 분할했음에도 불구하고, 모델 학습 시 전체 데이터인 fish_data로 모델을 학습시켰습니다. 이렇게 되면 평가용으로 완전히 숨겨놓아야 할 테스트 데이터(test_input)의 정보가 모델 학습에 유출(Data Leakage)되어, 객관적인 일반화 평가 성능을 측정할 수 없습니다. 따라서 (A) 단계에서 훈련 데이터인 train_input과 train_target으로 학습을 진행해야 마땅합니다."""
    },
    {
        "id": 5,
        "question": "Q5. 타이타닉 데이터(titanic.csv)를 불러와 탐색한 결과가 아래와 같을 때, 이 데이터의 머신러닝 문제 정의로 가장 올바르지 않은 것은?",
        "options": [
            "예측 대상(Label)은 생존 여부를 나타내는 Survived 컬럼이다",
            "이 문제의 유형은 정답 레이블이 있고 출력이 카테고리이므로 '지도학습의 분류(Classification)' 문제다",
            "승객의 이름(Name)이나 티켓 번호(Ticket)는 텍스트 형태이므로 전처리 없이 즉시 모델의 특성(Feature)으로 활용하기 가장 유망하다",
            "Age 컬럼은 177개의 결측값(Missing Value)이 존재하므로 평균값 대입 등의 전처리가 필요하다"
        ],
        "answer": 2,  # 0-indexed: 3번째 옵션이 정답
        "explanation": "승객의 이름(Name)이나 티켓 번호(Ticket)는 사람마다 거의 고유한 값을 갖는 비정형 텍스트 데이터입니다. 이러한 고유 비정형 텍스트는 인코딩 등의 특별하고 정교한 텍스트 전처리(자연어 처리) 과정을 거치지 않는 한 머신러닝 모델의 특성(Feature)으로 직접 넣을 수 없으며, 예측 성능 기여도도 낮습니다. 반면 성별(Sex)이나 객실 등급(Pclass) 등이 훨씬 유망한 Feature입니다."
    }
]

# 각 문제별 제출 상태 및 선택한 답변을 세션 스테이트에서 개별 관리합니다.
for idx, q in enumerate(quizzes):
    q_id = q["id"]
    submitted_key = f"q_{q_id}_submitted"
    choice_key = f"q_{q_id}_choice"
    
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
            key=f"radio_{q_id}",
            label_visibility="collapsed"
        )
        # 사용자가 클릭한 옵션의 인덱스를 실시간 저장
        st.session_state[choice_key] = q["options"].index(user_choice)
        
        # 개별 제출 버튼
        if st.button(f"📝 Q{q_id} 정답 제출", key=f"btn_{q_id}"):
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
        if st.button(f"🔄 Q{q_id} 다시 풀기", key=f"retry_{q_id}"):
            st.session_state[submitted_key] = False
            st.session_state[choice_key] = 0
            st.rerun()
            
    st.markdown("---")