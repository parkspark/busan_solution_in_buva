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
    # ---------------- 1. 🧠 머신러닝 기초 ----------------
    {
        "category": "1. 🧠 머신러닝 기초",
        "title": "지도 학습 (Supervised Learning)",
        "desc": "데이터와 <b>레이블(정답)</b>이 함께 주어지는 상태에서 학습하는 방식입니다. 인공지능에게 문제와 답안지를 같이 주고 공부시키는 것과 같습니다.",
        "badge": "학습 방식",
        "border_color": "#F59E0B",
        "keywords": ["지도학습", "정답", "레이블", "답안지"]
    },
    {
        "category": "1. 🧠 머신러닝 기초",
        "title": "특성(Feature) & 정답(Label)",
        "desc": "<b>특성(Feature)</b>은 문제를 풀기 위한 힌트(예: 물고기의 길이, 무게)이고, <b>정답(Label)</b>은 모델이 최종적으로 맞춰야 할 목표(예: 도미 1, 빙어 0)입니다.",
        "badge": "데이터",
        "border_color": "#F59E0B",
        "keywords": ["특성", "피처", "레이블", "정답", "힌트"]
    },
    {
        "category": "1. 🧠 머신러닝 기초",
        "title": "일반화 (Generalization)",
        "desc": "머신러닝 모델이 학습할 때 한 번도 보지 못한 <b>완전히 새로운 데이터(Test Data)</b>가 들어왔을 때도 정확하게 정답을 예측해 내는 능력입니다.",
        "badge": "핵심 목표",
        "border_color": "#F59E0B",
        "keywords": ["일반화", "generalization", "새로운데이터", "실전"]
    },

    # ---------------- 2. 🎯 무엇을 예측하고 평가할까? ----------------
    {
        "category": "2. 🎯 예측과 평가지표",
        "title": "회귀 (Regression)",
        "desc": "물고기의 무게, 주택 가격처럼 연속적인 흐름을 가진 <b>'연속된 숫자'</b>를 예측하는 머신러닝 알고리즘입니다.",
        "badge": "예측 방법론",
        "border_color": "#3B82F6",
        "keywords": ["회귀", "regression", "예측", "숫자"]
    },
    {
        "category": "2. 🎯 예측과 평가지표",
        "title": "결정계수 (R² Score) & MSE",
        "desc": "<b>R²</b>는 정답의 분산을 얼마나 설명하는지 나타내는 점수(1에 가까울수록 완벽)이며, <b>MSE(평균 제곱 오차)</b>는 예측값과 정답의 거리를 잰 오차(작을수록 정확)입니다.",
        "badge": "회귀 평가",
        "border_color": "#3B82F6",
        "keywords": ["결정계수", "r2", "mse", "오차"]
    },
    {
        "category": "2. 🎯 예측과 평가지표",
        "title": "오차 행렬 (Confusion Matrix)",
        "desc": "분류 모델이 맞았는지 틀렸는지를 <b>TP, FP, FN, TN의 4가지 방</b>으로 나누어, 어떤 부분에서 약점이 있는지 상세히 보여주는 성적표입니다.",
        "badge": "분류 평가",
        "border_color": "#3B82F6",
        "keywords": ["오차행렬", "혼동행렬", "confusion matrix", "tp", "fp", "fn", "tn"]
    },
    {
        "category": "2. 🎯 예측과 평가지표",
        "title": "정밀도(Precision) & 재현율(Recall)",
        "desc": "<b>정밀도</b>는 \"정답!\"이라고 외친 것 중 진짜 정답인 비율(신중함)이며, <b>재현율</b>은 숨은 진짜 정답 중 내가 찾아낸 비율(놓치지 않음)입니다. 이 둘의 조화평균이 <b>F1 Score</b>입니다.",
        "badge": "분류 평가",
        "border_color": "#3B82F6",
        "keywords": ["정밀도", "precision", "재현율", "recall", "f1"]
    },
    {
        "category": "2. 🎯 예측과 평가지표",
        "title": "ROC-AUC",
        "desc": "커트라인(임계값) 변화에 흔들리지 않고 모델이 클래스를 구별해 내는 <b>전반적인 랭킹(분별) 능력</b>을 0.5(랜덤)에서 1.0(완벽) 사이로 나타낸 종합 점수입니다.",
        "badge": "분류 평가",
        "border_color": "#3B82F6",
        "keywords": ["roc", "auc", "roc-auc", "종합점수"]
    },

    # ---------------- 3. 🔍 학습과 최적화 ----------------
    {
        "category": "3. 🔍 알고리즘과 최적화",
        "title": "경사 하강법 (Gradient Descent)",
        "desc": "산 안개 속에서 발끝의 경사만 보며 아래로 내려가듯, 오차의 기울기가 가파른 반대 방향으로 조금씩 이동하며 <b>가장 오차가 적은 최적의 가중치를 찾는 기법</b>입니다.",
        "badge": "최적화 기법",
        "border_color": "#10B981",
        "keywords": ["경사하강법", "gradient descent", "최적화"]
    },
    {
        "category": "3. 🔍 알고리즘과 최적화",
        "title": "파라미터 vs 하이퍼파라미터",
        "desc": "<b>파라미터</b>는 가중치(w)처럼 모델이 훈련하며 스스로 찾는 수치이고, <b>하이퍼파라미터</b>는 학습률(α)처럼 사람이 학습 시작 전에 직접 세팅해 주는 옵션 값입니다.",
        "badge": "설정 옵션",
        "border_color": "#10B981",
        "keywords": ["파라미터", "하이퍼파라미터", "가중치", "튜닝"]
    },
    {
        "category": "3. 🔍 알고리즘과 최적화",
        "title": "Sigmoid & Softmax",
        "desc": "로지스틱 회귀가 출력을 확률로 바꿀 때 씁니다. <b>Sigmoid</b>는 이진 분류(0~1 압축)에, <b>Softmax</b>는 다중 분류(모든 클래스 확률 합=1)에 사용됩니다.",
        "badge": "활성화 함수",
        "border_color": "#10B981",
        "keywords": ["시그모이드", "sigmoid", "소프트맥스", "softmax", "확률"]
    },
    {
        "category": "3. 🔍 알고리즘과 최적화",
        "title": "에포크(Epoch) & 조기 종료(Early Stopping)",
        "desc": "<b>에포크</b>는 훈련 데이터를 1회독 하는 것을 말합니다. 에포크를 늘리다 보면 과적합이 발생하는데, 검증 점수가 꺾이기 직전에 학습을 멈추는 것을 <b>조기 종료</b>라 합니다.",
        "badge": "학습 전략",
        "border_color": "#10B981",
        "keywords": ["에포크", "epoch", "조기종료", "early stopping", "반복"]
    },

    # ---------------- 4. 🚨 데이터 가공과 튜닝 ----------------
    {
        "category": "4. 🚨 데이터 가공과 주의점",
        "title": "특성 공학 (Feature Engineering)",
        "desc": "기초 데이터의 특성들을 곱하거나 거듭제곱하여 재조합함으로써, 인공지능이 <b>숨겨진 비선형 패턴을 더 부드럽고 똑똑하게 학습하도록 유도하는 기술</b>입니다.",
        "badge": "데이터 전처리",
        "border_color": "#8B5CF6",
        "keywords": ["특성공학", "피처엔지니어링", "재조합"]
    },
    {
        "category": "4. 🚨 데이터 가공과 주의점",
        "title": "데이터 누설 (Data Leakage)",
        "desc": "학습 중인 모델에 <b>미래의 정답(테스트 세트의 정보)이 슬쩍 섞여 학습이 오염되는 현상</b>입니다. 시험지를 미리 훔쳐본 것과 같아 실전 투입 시 치명적인 실패를 부릅니다.",
        "badge": "치명적 경고",
        "border_color": "#8B5CF6",
        "keywords": ["데이터누설", "누수", "leakage"]
    },
    {
        "category": "4. 🚨 데이터 가공과 주의점",
        "title": "결정 경계 (Decision Boundary)",
        "desc": "분류 모델이 합격과 불합격(0과 1)을 가르기 위해 공간상에 그어놓은 <b>기준선 (커트라인)</b>입니다. 기본적으로 확률 50%(z=0) 지점이 경계가 됩니다.",
        "badge": "기준선",
        "border_color": "#8B5CF6",
        "keywords": ["결정경계", "decision boundary", "커트라인"]
    },

    # ---------------- 5. 🤒 모델의 문제 현상과 해결 방법 ----------------
    {
        "category": "5. 🤒 모델의 문제 현상과 해결 방법",
        "title": "과적합 (High Variance)",
        "desc": "훈련 데이터의 특성에만 집중하여 암기해 버린 상태입니다. 연습문제는 100점이지만 처음 보는 테스트에서는 점수가 뚝 떨어집니다.",
        "badge": "문제 현상",
        "border_color": "#EC4899",
        "keywords": ["과적합", "overfitting", "high variance", "분산"]
    },
    {
        "category": "5. 🤒 모델의 문제 현상과 해결 방법",
        "title": "과소적합 (High Bias)",
        "desc": "모델이 너무 단순해서 훈련 데이터조차 제대로 패턴을 읽지 못한 미성숙한 상태입니다. 훈련과 테스트 점수 모두 낮게 나옵니다.",
        "badge": "문제 현상",
        "border_color": "#EC4899",
        "keywords": ["과소적합", "underfitting", "high bias", "편향"]
    },
    {
        "category": "5. 🤒 모델의 문제 현상과 해결 방법",
        "title": "규제 (Regularization)",
        "desc": "가중치의 크기가 무분별하게 커지는 것에 벌금(Penalty)을 부여하여, 수식을 부드럽게 만들고 과적합을 억누르는 <b>예방 치료법</b>입니다.",
        "badge": "해결 방법",
        "border_color": "#EC4899",
        "keywords": ["규제", "regularization", "벌금", "페널티"]
    },
    {
        "category": "5. 🤒 모델의 문제 현상과 해결 방법",
        "title": "릿지(Ridge) & 라쏘(Lasso) 회귀",
        "desc": "<b>릿지(L2)</b>는 가중치의 절댓값 제곱을 기준으로 벌금을 매겨 가중치를 골고루 축소시킵니다. <b>라쏘(L1)</b>는 가중치의 절댓값 합을 기준으로 벌금을 매겨 불필요한 특성을 아예 '0'으로 만듭니다.",
        "badge": "해결 방법",
        "border_color": "#EC4899",
        "keywords": ["릿지", "라쏘", "ridge", "lasso", "l1", "l2"]
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
