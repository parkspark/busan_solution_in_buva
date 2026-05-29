import streamlit as st
import streamlit.components.v1 as components
import os

st.title("📓 유방암 데이터셋 분석 - Jupyter Notebook 결과")
st.markdown("로지스틱 회귀를 이용한 유방암 데이터 분석 및 임계값 튜닝(FN 최소화) 실습 결과를 주피터 노트북 원본 그대로 확인합니다.")

# HTML 파일 경로 설정 (절대 경로 또는 상대 경로)
html_path = '../ml02/ml03-breast_cancer.html'
if not os.path.exists(html_path):
    html_path = r'c:\Users\금정산2-PC15\Desktop\busan_solution_in_buva\ML\ml02\ml03-breast_cancer.html'

if os.path.exists(html_path):
    # HTML 파일 읽기
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        
    # 사용자가 프레임 높이를 조절할 수 있는 슬라이더 제공
    viewer_height = st.slider("노트북 뷰어 높이 조절 (px)", min_value=500, max_value=2000, value=800, step=100)
    
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
    st.info("프로젝트의 `ML/ml02/ml03-breast_cancer.ipynb` 파일을 HTML로 정상 변환했는지 다시 확인해 주세요.")

st.divider()

st.markdown(""" 
### 핵심 요약

**의료 도메인에서의 분류 평가지표**
질병 진단과 같은 도메인에서는 암 환자를 정상으로 오진하는 것(False Negative, FN)이 생명에 직결되므로 매우 치명적입니다. 따라서 단순히 `Accuracy(정확도)`가 높은 모델보다 **`Recall(재현율)`**이 높은 모델을 선호합니다.

**임계값(Threshold) 조정의 효과**
로지스틱 회귀 모델의 기본 임계값은 50%(0.5)입니다. 이 임계값을 30%(0.3)로 낮추게 되면 모델은 '조금만 암으로 의심되어도' 암이라고 진단하게 됩니다. 
* **장점:** 놓치는 암 환자 수(FN)가 크게 줄어들며, 이로 인해 전체 암 환자 중 암으로 정확히 진단된 비율인 **Recall이 100%에 가깝게 상승**합니다.
* **단점:** 단순 종양인 환자를 암으로 오해하는 거짓 알람(False Positive, FP)이 증가하여 **Precision(정밀도)은 하락**하는 트레이드오프(Trade-off)가 발생합니다.
""")
