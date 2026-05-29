import streamlit as st

# Streamlit 페이지 설정
st.set_page_config(
    page_title="통합 머신러닝 핵심 개념 사전",
    page_icon="📖",
    layout="wide"
)

# 커스텀 CSS 스타일링 (다크 테마 최적화 및 프리미엄 카드 디자인)
st.markdown(
    """
    <style>
    /* 제목 및 배너 스타일 */
    .title-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* 카테고리 헤더 스타일 */
    .category-header {
        font-size: 22px;
        font-weight: 700;
        color: #60A5FA;
        border-bottom: 2px solid #3B82F6;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    
    /* 프리미엄 카드 스타일 */
    .concept-card {
        background-color: #1E293B;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 90%;
    }
    .concept-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.25);
    }
    
    /* 카드 제목 및 뱃지 */
    .card-header-wrapper {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
    }
    .card-badge {
        background-color: #2563EB;
        color: white;
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* 카드 설명 */
    .card-desc {
        color: #E2E8F0;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 1. 메인 타이틀 배너
st.markdown(
    """
    <div class="title-banner">
        <h1 style='margin: 0; font-size: 32px;'>📖 통합 머신러닝 핵심 개념 사전</h1>
        <p style='margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;'>
            지도학습 기초부터 분류 알고리즘까지! 지금까지 배운 모든 핵심 용어를 한 곳에 모았습니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# 2. 용어 데이터 정의
vocab_data = [
    # ---------------- 1. 기초 ----------------
    {
        "category": "🧠 기초",
        "title": "지도학습",
        "desc": "데이터와 정답이 함께 주어지는 학습 방식입니다. 인공지능에게 문제와 답안지를 같이 주고 공부시키는 것과 같습니다.",
        "badge": "학습방식",
        "border_color": "#F59E0B",
        "keywords": ["지도학습", "정답", "레이블", "답안지"]
    },
    {
        "category": "🧠 기초",
        "title": "특성",
        "desc": "문제를 풀기 위한 힌트를 의미하며, 머신러닝 모델의 입력 데이터로 사용됩니다.",
        "badge": "데이터",
        "border_color": "#F59E0B",
        "keywords": ["특성", "피처", "힌트"]
    },
    {
        "category": "🧠 기초",
        "title": "정답",
        "desc": "모델이 최종적으로 맞춰야 할 목표를 의미하며, 지도학습의 필수 요소입니다.",
        "badge": "데이터",
        "border_color": "#F59E0B",
        "keywords": ["레이블", "정답", "목표"]
    },
    {
        "category": "🧠 기초",
        "title": "일반화",
        "desc": "모델이 한 번도 보지 못한 새로운 데이터에 대해서도 정확하게 예측해 내는 능력입니다.",
        "badge": "목표",
        "border_color": "#F59E0B",
        "keywords": ["일반화", "generalization", "실전"]
    },
    {
        "category": "🧠 기초",
        "title": "회귀",
        "desc": "주택 가격이나 온도의 변화처럼 연속적인 숫자의 흐름을 예측하는 알고리즘입니다.",
        "badge": "예측",
        "border_color": "#F59E0B",
        "keywords": ["회귀", "regression", "예측", "숫자"]
    },
    {
        "category": "🧠 기초",
        "title": "분류",
        "desc": "데이터가 어떤 범주에 속하는지 정해진 클래스 중 하나를 예측하는 알고리즘입니다.",
        "badge": "예측",
        "border_color": "#F59E0B",
        "keywords": ["분류", "classification", "예측", "클래스"]
    },

    # ---------------- 2. 평가지표 ----------------
    {
        "category": "🎯 평가지표",
        "title": "결정계수",
        "desc": "회귀 모델의 성능 지표로, 정답의 분산을 얼마나 잘 설명하는지 나타냅니다. 1에 가까울수록 좋습니다.",
        "badge": "회귀",
        "border_color": "#3B82F6",
        "keywords": ["결정계수", "r2", "회귀"]
    },
    {
        "category": "🎯 평가지표",
        "title": "오차",
        "desc": "예측값과 정답 간의 차이를 의미합니다. 평균 제곱 오차(MSE) 등이 사용되며 작을수록 정확합니다.",
        "badge": "회귀",
        "border_color": "#3B82F6",
        "keywords": ["오차", "mse", "손실"]
    },
    {
        "category": "🎯 평가지표",
        "title": "오차행렬",
        "desc": "분류 모델의 예측 결과를 정답과 오답(TP, FP, FN, TN) 4가지로 상세히 나누어 보여주는 표입니다.",
        "badge": "분류",
        "border_color": "#3B82F6",
        "keywords": ["오차행렬", "혼동행렬", "confusion matrix", "tp", "fp"]
    },
    {
        "category": "🎯 평가지표",
        "title": "정밀도",
        "desc": "모델이 정답이라고 예측한 것 중에서 실제 정답인 비율입니다. 모델의 신중함을 나타냅니다.",
        "badge": "분류",
        "border_color": "#3B82F6",
        "keywords": ["정밀도", "precision", "분류"]
    },
    {
        "category": "🎯 평가지표",
        "title": "재현율",
        "desc": "실제 정답 중에서 모델이 정확하게 찾아낸 비율입니다. 정답을 놓치지 않는 능력을 나타냅니다.",
        "badge": "분류",
        "border_color": "#3B82F6",
        "keywords": ["재현율", "recall", "분류"]
    },
    {
        "category": "🎯 평가지표",
        "title": "F1스코어",
        "desc": "정밀도와 재현율의 조화평균으로, 데이터가 불균형할 때 사용하기 좋은 종합 평가 점수입니다.",
        "badge": "분류",
        "border_color": "#3B82F6",
        "keywords": ["f1", "f1스코어", "조화평균"]
    },

    # ---------------- 3. 최적화 ----------------
    {
        "category": "🔍 최적화",
        "title": "경사하강법",
        "desc": "오차의 기울기가 가파른 반대 방향으로 조금씩 이동하며 최적의 가중치를 찾는 기법입니다.",
        "badge": "알고리즘",
        "border_color": "#10B981",
        "keywords": ["경사하강법", "gradient descent", "최적화"]
    },
    {
        "category": "🔍 최적화",
        "title": "파라미터",
        "desc": "가중치나 편향처럼 모델이 훈련 데이터를 통해 스스로 학습하고 갱신해 나가는 수치입니다.",
        "badge": "설정",
        "border_color": "#10B981",
        "keywords": ["파라미터", "가중치", "학습"]
    },
    {
        "category": "🔍 최적화",
        "title": "하이퍼파라미터",
        "desc": "학습률이나 트리의 깊이처럼 모델 학습 전에 사람이 직접 설정해 주어야 하는 옵션 값입니다.",
        "badge": "설정",
        "border_color": "#10B981",
        "keywords": ["하이퍼파라미터", "튜닝", "설정"]
    },
    {
        "category": "🔍 최적화",
        "title": "시그모이드",
        "desc": "로지스틱 회귀에서 출력을 0과 1 사이의 확률로 압축하여 이진 분류를 수행하는 활성화 함수입니다.",
        "badge": "함수",
        "border_color": "#10B981",
        "keywords": ["시그모이드", "sigmoid", "이진분류", "확률"]
    },
    {
        "category": "🔍 최적화",
        "title": "소프트맥스",
        "desc": "다중 분류에서 여러 클래스의 출력값을 합쳐 1이 되는 확률 분포로 변환해 주는 활성화 함수입니다.",
        "badge": "함수",
        "border_color": "#10B981",
        "keywords": ["소프트맥스", "softmax", "다중분류", "확률"]
    },
    {
        "category": "🔍 최적화",
        "title": "에포크",
        "desc": "전체 훈련 데이터셋을 모델이 한 번 모두 학습하는 1회독 과정을 의미합니다.",
        "badge": "학습",
        "border_color": "#10B981",
        "keywords": ["에포크", "epoch", "반복"]
    },

    # ---------------- 4. 데이터 ----------------
    {
        "category": "🚨 데이터",
        "title": "특성공학",
        "desc": "기존 데이터를 조합하거나 변형하여 모델이 숨겨진 패턴을 더 잘 학습하도록 유도하는 기술입니다.",
        "badge": "전처리",
        "border_color": "#8B5CF6",
        "keywords": ["특성공학", "피처엔지니어링", "재조합"]
    },
    {
        "category": "🚨 데이터",
        "title": "데이터누설",
        "desc": "모델 학습 과정에 테스트 세트의 정보가 섞여 들어가 실전 성능을 망치는 치명적인 오류입니다.",
        "badge": "주의",
        "border_color": "#8B5CF6",
        "keywords": ["데이터누설", "누수", "leakage"]
    },
    {
        "category": "🚨 데이터",
        "title": "결정경계",
        "desc": "분류 모델이 클래스를 나누기 위해 데이터 공간 상에 긋는 기준선 또는 커트라인입니다.",
        "badge": "개념",
        "border_color": "#8B5CF6",
        "keywords": ["결정경계", "decision boundary", "커트라인"]
    },
    {
        "category": "🚨 데이터",
        "title": "과적합",
        "desc": "모델이 훈련 데이터에만 지나치게 맞춰져 새로운 데이터에 대한 예측력이 떨어지는 현상입니다.",
        "badge": "문제",
        "border_color": "#8B5CF6",
        "keywords": ["과적합", "overfitting", "high variance"]
    },
    {
        "category": "🚨 데이터",
        "title": "과소적합",
        "desc": "모델이 너무 단순하여 훈련 데이터의 기본적인 패턴조차 제대로 학습하지 못한 현상입니다.",
        "badge": "문제",
        "border_color": "#8B5CF6",
        "keywords": ["과소적합", "underfitting", "high bias"]
    },
    {
        "category": "🚨 데이터",
        "title": "규제",
        "desc": "가중치가 너무 커지지 않도록 벌칙을 주어 모델을 단순화하고 과적합을 방지하는 기법입니다.",
        "badge": "해결",
        "border_color": "#8B5CF6",
        "keywords": ["규제", "regularization", "페널티"]
    },

    # ---------------- 5. 앙상블 ----------------
    {
        "category": "🌳 앙상블",
        "title": "결정트리",
        "desc": "스무고개처럼 조건에 따라 데이터를 분기하며 정답을 찾아가는 직관적인 형태의 알고리즘입니다.",
        "badge": "모델",
        "border_color": "#059669",
        "keywords": ["결정트리", "decision tree", "스무고개", "노드"]
    },
    {
        "category": "🌳 앙상블",
        "title": "불순도",
        "desc": "데이터가 얼마나 섞여 있는지를 나타내는 지표로, 트리는 불순도를 최소화하는 방향으로 분기합니다.",
        "badge": "지표",
        "border_color": "#059669",
        "keywords": ["지니", "gini", "엔트로피", "불순도"]
    },
    {
        "category": "🌳 앙상블",
        "title": "가지치기",
        "desc": "결정트리가 너무 깊어져 과적합되는 것을 막기 위해 트리의 최대 깊이나 분기 조건을 제한하는 기법입니다.",
        "badge": "최적화",
        "border_color": "#059669",
        "keywords": ["가지치기", "max_depth", "제한"]
    },
    {
        "category": "🌳 앙상블",
        "title": "배깅",
        "desc": "데이터를 무작위로 복원 추출하여 여러 모델을 병렬 학습시킨 후 다수결로 결합하는 앙상블 기법입니다.",
        "badge": "기법",
        "border_color": "#059669",
        "keywords": ["배깅", "bagging", "복원추출", "다수결"]
    },
    {
        "category": "🌳 앙상블",
        "title": "랜덤포레스트",
        "desc": "다수의 결정트리를 모아 숲을 이루는 배깅 기반 모델로, 특성도 무작위로 선택하여 과적합을 방지합니다.",
        "badge": "모델",
        "border_color": "#059669",
        "keywords": ["랜덤포레스트", "random forest", "rf"]
    },
    {
        "category": "🌳 앙상블",
        "title": "부스팅",
        "desc": "이전 모델의 오답을 다음 모델이 집중적으로 보완하며 순차적으로 학습하는 강력한 앙상블 기법입니다.",
        "badge": "기법",
        "border_color": "#059669",
        "keywords": ["부스팅", "boosting", "순차적", "잔차"]
    },

    # ---------------- 6. 검증 ----------------
    {
        "category": "⚙️ 검증",
        "title": "교차검증",
        "desc": "데이터를 여러 등분으로 나누어 번갈아 가며 검증을 수행하여 평가의 신뢰도를 높이는 방법입니다.",
        "badge": "평가",
        "border_color": "#D97706",
        "keywords": ["k-fold", "교차검증", "cv", "kfold"]
    },
    {
        "category": "⚙️ 검증",
        "title": "그리드서치",
        "desc": "지정된 하이퍼파라미터의 모든 조합을 전수 조사하여 최적의 모델 설정을 찾아내는 튜닝 도구입니다.",
        "badge": "튜닝",
        "border_color": "#D97706",
        "keywords": ["gridsearch", "gridsearchcv", "그리드서치", "전수조사"]
    },
    {
        "category": "⚙️ 검증",
        "title": "랜덤서치",
        "desc": "파라미터 조합이 너무 많을 때 무작위로 일부만 샘플링하여 빠르게 최적 설정을 탐색하는 튜닝 도구입니다.",
        "badge": "튜닝",
        "border_color": "#D97706",
        "keywords": ["randomizedsearchcv", "randomsearch", "랜덤서치"]
    }
]

# 3. 실시간 인터랙티브 개념 검색 및 필터링 기능
st.markdown("### 🔍 개념 검색")
search_query = st.text_input(
    "알고 싶은 용어가 있으신가요? 키워드 또는 알파벳을 입력해 보세요! (예: 회귀, 오차, 과적합, L1)",
    ""
).strip().lower()

# 검색 필터링 logic
filtered_data = []
if search_query:
    for item in vocab_data:
        # 제목, 설명, 뱃지, 키워드 검색 일치 확인
        match = (
            search_query in item["title"].lower() or 
            search_query in item["desc"].lower() or 
            search_query in item["badge"].lower() or
            any(search_query in kw for kw in item["keywords"])
        )
        if match:
            filtered_data.append(item)
else:
    filtered_data = vocab_data

# 검색 결과 멘트
if search_query:
    st.write(f"🔎**{search_query}** 에 대한 검색 결과가 총 **{len(filtered_data)}건** 검색되었습니다.")
    if len(filtered_data) == 0:
        st.warning("앗! 찾으시는 단어와 일치하는 개념이 보이지 않아요. 다른 검색어로 찾아볼까요?")

st.write("---")

# 4. 필터링된 결과 카드 렌더링
if len(filtered_data) > 0:
    # 카테고리 순서를 유지하기 위해 직접 정렬
    categories = []
    for item in filtered_data:
        if item["category"] not in categories:
            categories.append(item["category"])
    
    for cat in categories:
        st.markdown(f'<div class="category-header">{cat}</div>', unsafe_allow_html=True)
        
        # 해당 카테고리에 속하는 아이템만 렌더링
        cat_items = [item for item in filtered_data if item["category"] == cat]
        
        # 2열(2 columns) 레이아웃으로 카드를 깔끔하고 트렌디하게 렌더링
        col1, col2 = st.columns(2)
        
        for idx, item in enumerate(cat_items):
            card_html = f"""
            <div class="concept-card" style="border-left-color: {item['border_color']};">
                <div class="card-header-wrapper">
                    <h3 class="card-title">{item['title']}</h3>
                    <span class="card-badge" style="background-color: {item['border_color']};">{item['badge']}</span>
                </div>
                <p class="card-desc">{item['desc']}</p>
            </div>
            """
            if idx % 2 == 0:
                with col1:
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                with col2:
                    st.markdown(card_html, unsafe_allow_html=True)
else:
    # 전체 보기 안내
    st.info("💡 위의 검색창을 비우시면 전체 개념 사전을 확인하실 수 있습니다.")
