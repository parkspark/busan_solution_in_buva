import streamlit as st
# Streamlit 페이지 설정
st.set_page_config(
    page_title="한 눈에 보는 머신러닝 핵심 개념 사전",
    page_icon="📚",
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
        <h1 style='margin: 0; font-size: 32px;'>📚 한 눈에 보는 머신러닝 핵심 개념 사전</h1>
        <p style='margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;'>
            머신러닝이 어떻게 숫자를 예측하고 스스로 똑똑해지는지 쉽게 이해하기 위한 필수 용어 정리집입니다.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# 2. 용어 데이터 정의
vocab_data = [
    # 1. 무엇을 예측하고 평가할까?
    {
        "category": "🎯 무엇을 예측하고 평가할까?",
        "title": "회귀 (Regression)",
        "desc": "물고기의 무게, 주택 가격처럼 연속적인 흐름을 가진 <b>'연속된 숫자'</b>를 예측하는 핵심 머신러닝 알고리즘 학습 방법이에요.",
        "badge": "예측 방법론",
        "border_color": "#3B82F6",
        "keywords": ["회귀", "regression", "예측", "숫자", "물무게"]
    },
    {
        "category": "🎯 무엇을 예측하고 평가할까?",
        "title": "결정계수 (R² Score)",
        "desc": "모델이 실제 정답 데이터의 분산(흩어짐 정도)을 얼마나 완벽하게 설명하는지 나타내는 성능 점수표예요. <b>1점에 가까울수록 정답을 완벽하게 맞히는 신뢰도 높은 모델</b>임을 뜻해요.",
        "badge": "평가 지표",
        "border_color": "#3B82F6",
        "keywords": ["결정계수", "r2", "결정 계수", "점수", "설명력"]
    },
    {
        "category": "🎯 무엇을 예측하고 평가할까?",
        "title": "평균 제곱 오차 (MSE - Mean Squared Error)",
        "desc": "인공지능의 예측값과 실제 정답이 얼마나 틀렸는지(오차)를 각각 제곱한 뒤, 그 값들을 모두 합해 평균을 낸 오차 지표예요. <b>값이 작을수록 정답과의 거리(오차)가 좁고 정확함</b>을 의미해요.",
        "badge": "오차 지표",
        "border_color": "#3B82F6",
        "keywords": ["평균제곱오차", "mse", "오차", "평균 제곱", "에러"]
    },
    
    # 2. 어떻게 학습하고 최적화할까?
    {
        "category": "🔍 어떻게 학습하고 최적화할까?",
        "title": "경사 하강법 (Gradient Descent)",
        "desc": "산 안개 속에서 길을 잃었을 때 발끝의 경사만을 보며 아래로 내려가듯, <b>오차 함수의 기울기가 가장 가파른 반대 방향으로 조금씩 이동하며 오차가 가장 적은 최적의 가중치를 찾아가는 기법</b>이에요.",
        "badge": "최적화 기법",
        "border_color": "#10B981",
        "keywords": ["경사하강법", "경사", "기울기", "가장 적은", "최적화"]
    },
    {
        "category": "🔍 어떻게 학습하고 최적화할까?",
        "title": "학습률 (α - Learning Rate)",
        "desc": "경사 하강법을 진행할 때 한 걸음에 <b>얼마만큼씩 가중치를 보정해 나갈지 결정하는 보폭(움직임의 크기)</b>이에요. 보폭이 너무 크면 최적점을 지나치고, 너무 작으면 산에서 내려오는 데 하루 종일 걸릴 수 있어요.",
        "badge": "하이퍼파라미터",
        "border_color": "#10B981",
        "keywords": ["학습률", "learning rate", "알파", "alpha", "보폭", "크기"]
    },
    {
        "category": "🔍 어떻게 학습하고 최적화할까?",
        "title": "파라미터 vs 하이퍼파라미터 (Parameter vs Hyperparameter)",
        "desc": "<ul><li><b>모델 파라미터:</b> 모델이 데이터를 학습하면서 <b>스스로 최적값을 찾아내는 내재적 수치</b>예요 (예: 회귀 선형 방정식의 가중치 w, 절편 b).</li><li><b>하이퍼파라미터:</b> 모델이 스스로 정할 수 없어, <b>사람이 훈련 시작 전에 직접 튜닝하여 쥐여줘야 하는 옵션 값</b>이에요 (예: 규제 강도 alpha, k-NN 알고리즘의 이웃 개수 K).</li></ul>",
        "badge": "설정 옵션",
        "border_color": "#10B981",
        "keywords": ["파라미터", "하이퍼파라미터", "가중치", "w", "b", "이웃", "옵션"]
    },
    # 3. 데이터 가공과 주의점
    {
        "category": "🚨 데이터 가공과 주의점",
        "title": "특성 공학 (Feature Engineering)",
        "desc": "가지고 있는 기초 데이터의 변수(특성)들을 서로 곱하거나 거듭제곱하는 등 다양하게 재조합하여, <b>인공지능 모델이 데이터 이면의 숨겨진 비선형 패턴을 더 부드럽고 똑똑하게 학습할 수 있도록 유도하는 기술</b>이에요.",
        "badge": "데이터 전처리",
        "border_color": "#F59E0B",
        "keywords": ["특성공학", "특성 공학", "피처", "재조합", "패턴"]
    },
    {
        "category": "🚨 데이터 가공과 주의점",
        "title": "데이터 누설 (Data Leakage)",
        "desc": "학습 중인 모델에 <b>미래의 정답(테스트 데이터 세트의 정보)이 슬쩍 들어가 학습이 오염되는 위험천만한 현상</b>이에요. 시험지를 미리 훔쳐보고 시험을 친 것과 같아서, 훈련 점수는 만점이 나와도 실전 검증(실제 투입) 시 엉망진창인 결과를 내게 만들어요.",
        "badge": "치명적 경고",
        "border_color": "#EF4444",
        "keywords": ["데이터누설", "데이터 누설", "누수", "leakage", "테스트"]
    },
    # 4. 모델이 걸릴 수 있는 병과 치료법
    {
        "category": "🤒 모델이 걸릴 수 있는 병과 치료법",
        "title": "과적합 (Overfitting)",
        "desc": "모델이 주어진 훈련 데이터의 특성에만 아주 뼈저리게 집중하고 암기한 탓에, <b>처음 마주하는 시험(테스트 데이터)에서는 엉터리 성적표를 받아오는 문제 상태</b>를 말해요. 융통성이 부족한 고지식한 모델이 된 셈이죠.",
        "badge": "모델 질병",
        "border_color": "#8B5CF6",
        "keywords": ["과적합", "overfitting", "질병", "암기", "훈련만"]
    },
    {
        "category": "🤒 모델이 걸릴 수 있는 병과 치료법",
        "title": "과소적합 (Underfitting)",
        "desc": "모델의 표현력(구조)이 데이터의 난이도에 비해 너무 심플하고 단순해서, <b>기본 연습문제(훈련 데이터)조차도 제대로 패턴을 읽지 못하고 전체적으로 낮은 점수를 겉도는 미성숙한 상태</b>를 말해요.",
        "badge": "모델 질병",
        "border_color": "#8B5CF6",
        "keywords": ["과소적합", "underfitting", "질병", "단순", "낮은점수"]
    },
    {
        "category": "🤒 모델이 걸릴 수 있는 병과 치료법",
        "title": "규제 (Regularization)",
        "desc": "인공지능 모델이 수식을 억지로 구부려 훈련 데이터를 강제로 외우지(과적합) 못하게 하도록, <b>가중치의 크기가 무럭무럭 자라는 것에 일정 벌금(Penalty)을 부여하여 수식을 부드럽고 간결하게 억누르는 예방 치료법</b>이에요.",
        "badge": "치료 비법",
        "border_color": "#10B981",
        "keywords": ["규제", "regularization", "벌금", "페널티", "가중치 억제"]
    },
    {
        "category": "🤒 모델이 걸릴 수 있는 병과 치료법",
        "title": "릿지 회귀 (Ridge Regression)",
        "desc": "선형 회귀 가중치들의 <b>절댓값 제곱합(L2 페널티)을 기준으로 벌금을 부여하는 일반적인 규제 방법</b>이에요. 특성들의 계수를 0에 가깝도록 아주 잘게 골고루 작게 축소시켜 부드러운 예측 곡선을 만들어내요.",
        "badge": "릿지(L2)",
        "border_color": "#EC4899",
        "keywords": ["릿지", "ridge", "l2", "제곱합", "가중치축소"]
    },
    {
        "category": "🤒 모델이 걸릴 수 있는 병과 치료법",
        "title": "라쏘 회귀 (Lasso Regression)",
        "desc": "가중치들의 <b>절댓값의 합(L1 페널티)을 기준으로 벌금을 부여하는 규제 방법</b>이에요. 중요도가 몹시 떨어지는 불필요한 특성의 계수는 <b>아예 칼같이 '0'으로 만들어버려 자동 특성 선택 효과</b>를 선사해요.",
        "badge": "라쏘(L1)",
        "border_color": "#EC4899",
        "keywords": ["라쏘", "lasso", "l1", "절댓값", "0으로"]
    }
]
# 3. 실시간 인터랙티브 개념 검색 및 필터링 기능
st.markdown("### 🔍 실시간 개념 검색 돋보기")
search_query = st.text_input(
    "알고 싶은 용어가 있으신가요? 키워드 또는 알파벳을 입력해 보세요! (예: 회귀, alpha, 과적합, L1)",
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
    st.write(f"🔎 **'{search_query}'**에 대한 검색 결과가 총 **{len(filtered_data)}건** 검색되었습니다.")
    if len(filtered_data) == 0:
        st.warning("앗! 찾으시는 단어와 일치하는 개념이 보이지 않아요. 다른 검색어로 찾아볼까요?")
st.write("---")
# 4. 필터링된 결과 카드 렌더링
if len(filtered_data) > 0:
    # 카테고리별 그룹화
    categories = sorted(list(set(item["category"] for item in filtered_data)))
    
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