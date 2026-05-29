import streamlit as st
import streamlit.components.v1 as components
import os

# Streamlit 페이지 설정
st.set_page_config(
    page_title="캘리포니아 주택가격 분석 Jupyter Notebook",
    page_icon="📓",
    layout="wide"
)

# 헤더 디자인
st.title("📓 캘리포니아 주택가격 분석 Jupyter Notebook")

st.divider()

# HTML 파일의 절대 경로 설정
# ML02_06_califonia.py 파일의 디렉토리는 ML/ml01/ML02 이므로, 상위 폴더의 상위 폴더인 ML 폴더 아래 ml02 폴더에 위치한 HTML 파일을 읽습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(current_dir, "..", "..", "ml02", "ml02-california_housing.html")

if os.path.exists(html_path):
    # HTML 파일 읽기
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # 상단 컨트롤 패널 (프리미엄 UI/UX)
    col_ctrl, col_download = st.columns([3, 1])
    
    with col_ctrl:
        st.info("💡 아래 화면에서 스크롤을 내려 Jupyter Notebook의 전체 코드와 시각화 실행 결과를 자세히 확인하실 수 있습니다.")
        # 사용자가 화면 해상도에 맞게 iframe 높이를 동적으로 조절할 수 있도록 슬라이더 제공
        viewer_height = st.slider(
            "📐 노트북 뷰어 높이 조절 (픽셀 단위)",
            min_value=400,
            max_value=3000,
            value=1200,
            step=100
        )
        
    with col_download:
        st.write("#### 💾 파일 다운로드")
        # HTML 파일을 브라우저에서 직접 다운로드하여 새 탭에서 열어볼 수 있도록 지원
        st.download_button(
            label="🌐 HTML 원본 파일 다운로드",
            data=html_content,
            file_name="ml02-california_housing.html",
            mime="text/html",
            use_container_width=True
        )
    
    st.write("---")
    
    # 쥬피터 노트북 HTML을 아름다운 카드 형태의 프레임 내에 렌더링
    st.markdown(
        """
        <style>
        .notebook-container {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            padding: 5px;
            background-color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # HTML components를 이용하여 IFrame으로 안전하게 렌더링
    st.markdown('<div class="notebook-container">', unsafe_allow_html=True)
    components.html(html_content, height=viewer_height, scrolling=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error(f"⚠️ HTML 결과 파일을 찾을 수 없습니다. 파일이 존재하지 않거나 경로가 잘못되었습니다.\n\n예상 경로: `{html_path}`")
    st.info("프로젝트 루트의 `ML/ml02/ml02-california_housing.ipynb` 파일을 HTML로 정상 변환했는지 다시 확인해 주세요.")




st.markdown(""" 
alpha가 10일 때 모델의 예측 성능이 가장 우수.

일반적으로 하이퍼파라미터 튜닝(예: GridSearchCV, RidgeCV 등) 시 여러 개의 alpha 후보값(예: 0.01, 0.1, 1, 10, 100 등)을 설정하고 각각의 성능을 테스트합니다.

여기서 alpha가 10으로 선택되었다는 것은 다음과 같은 의미를 가집니다.

최적의 오차율(성능 지표): 검증 데이터 세트를 기준으로 테스트한 후보들 중, alpha가 10일 때 평균 제곱 오차(MSE)가 가장 낮거나 결정 계수(R² Score)가 가장 높게 나타났습니다.

과적합(Overfitting) 방지와 일반화: alpha는 모델의 가중치(Coefficient)가 너무 커지지 않도록 벌점(Penalty)을 부여하는 수치입니다. 
10이라는 수치가 캘리포니아 주택 가격 데이터셋의 특성을 학습할 때, 훈련 데이터에만 과도하게 맞춰지는 과적합을 방지하면서도 새로운 데이터에 대한 예측력을 가장 높게 유지하는 최적의 균형점(Trade-off)이었음을 의미합니다.

""")