import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# 한글 폰트 설정 (Windows 기준 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="OOB(Out-Of-Bag) 란?", page_icon="🎒", layout="wide")

st.title("🎒 OOB(Out-Of-Bag) 평가 완벽 이해하기")
st.write("랜덤 포레스트(Random Forest) 모델을 학습할 때, 별도의 검증 데이터(Validation Set) 없이도 모델의 성능을 평가할 수 있는 마법 같은 방법인 **OOB(Out-Of-Bag)**에 대해 알아봅니다.")

st.divider()

st.header("1. OOB(Out-Of-Bag)란 무엇인가요? 🤔")

st.markdown("""
**OOB(Out-Of-Bag)**는 단어 뜻 그대로 **'가방(Bag) 밖에 있는'** 데이터입니다.

랜덤 포레스트는 여러 개의 결정 트리(Decision Tree)를 만들기 위해 전체 데이터에서 무작위로 데이터를 뽑아(복원 추출, Bootstrapping) 각각의 트리를 학습시킵니다. 
이때, 수학적으로 **어떤 트리에도 뽑히지 않고 남아있는 약 36.8%의 데이터**가 반드시 생기게 되는데, 이것을 바로 **OOB 샘플(Out-Of-Bag Sample)**이라고 부릅니다.

- **Bagging (Bootstrap Aggregating)**: 데이터를 무작위로 복원 추출하여 여러 트리를 만드는 과정
- **In-Bag Data**: 특정 트리를 학습하는 데 사용되도록 뽑힌 데이터
- **Out-Of-Bag (OOB) Data**: 특정 트리를 학습할 때 한 번도 뽑히지 않아 **학습에 사용되지 않은 나머지 데이터**
""")

st.info("💡 **핵심 요약:** OOB 데이터는 해당 트리를 학습할 때 전혀 사용되지 않았으므로, 이를 마치 **테스트 데이터(Test Data)처럼 사용하여 모델의 성능을 평가**할 수 있습니다. 이렇게 측정한 점수를 **OOB 평가(OOB Evaluation)** 또는 **OOB 점수(OOB Score)**라고 합니다.")

st.divider()

st.header("2. 일상 생활로 비유해볼까요? 🍪")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 👩‍🍳 쿠키 굽기 비유 (데이터 관점)
    - **전체 반죽 (전체 데이터)**: 쿠키 반죽 한 통이 있습니다.
    - **요리사 1 (트리 1)**: 반죽에서 한 움큼(무작위 추출) 떼어내어 초코 쿠키를 굽습니다.
    - **요리사 2 (트리 2)**: 다시 전체 반죽에서 한 움큼 떼어내어 아몬드 쿠키를 굽습니다.
    - **OOB 데이터 (남은 반죽)**: 여러 요리사가 반죽을 떼어갔지만, **어떤 요리사도 가져가지 않아 그대로 남은 반죽**이 있습니다.
    
    👉 요리사들이 만든 쿠키 레시피가 훌륭한지 평가하고 싶을 때, 새로운 반죽을 사올 필요 없이 **아무도 건드리지 않았던 남은 반죽(OOB)**으로 쿠키를 구워 맛을 평가하면 됩니다!
    """)

with col2:
    st.markdown("""
    ### 📝 모의고사 비유 (학습 관점)
    - **전체 문제 은행 (전체 데이터)**: 총 100문제가 있습니다.
    - **학생 A (트리 1)**: 100문제 중 무작위로 60문제를 뽑아 공부합니다.
    - **학생 B (트리 2)**: 100문제 중 무작위로 60문제를 뽑아 공부합니다.
    - **OOB 문제 (안 푼 문제)**: 학생 A가 **한 번도 본 적 없는 40문제**가 있습니다. 이것이 학생 A의 OOB 데이터입니다.
    
    👉 학생 A의 진짜 실력을 테스트하기 위해 새로운 시험지를 만들 필요 없이, **학생 A가 공부할 때 보지 못했던 40문제(OOB)**를 풀게 하여 점수(OOB Score)를 매기면 됩니다!
    """)

st.divider()

st.header("3. 파이썬 코드 시각화로 확인하기 📊")
st.write("유방암(Breast Cancer) 진단 데이터를 사용하여, 랜덤 포레스트에서 트리의 개수(`n_estimators`)가 늘어날 때 OOB 점수가 어떻게 변화하는지 시각화해 보겠습니다.")

with st.echo():
    # 1. 데이터 준비
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # 2. OOB 점수를 저장할 리스트와 탐색할 트리의 개수 설정
    oob_scores = []
    n_estimators_range = range(15, 151, 5) # 트리 개수 15개부터 150개까지 5개씩 증가
    
    # 3. 모델 학습 및 OOB 점수 기록
    for n in n_estimators_range:
        # 중요! oob_score=True 로 설정해야 OOB 평가를 수행합니다.
        rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
        rf.fit(X, y) # 검증 데이터 분리 없이 전체 데이터(X, y)로 바로 학습!
        
        # 모델에 저장된 OOB 점수(.oob_score_)를 리스트에 추가
        oob_scores.append(rf.oob_score_)

# 4. 시각화 (Matplotlib)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(n_estimators_range, oob_scores, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=7)

# 차트 꾸미기
ax.set_title('🌳 트리의 개수(n_estimators)에 따른 OOB 점수(정확도) 변화', fontsize=16, pad=20, fontweight='bold')
ax.set_xlabel('트리의 개수 (n_estimators)', fontsize=13)
ax.set_ylabel('OOB 점수 (정확도)', fontsize=13)
ax.grid(True, linestyle='--', alpha=0.7)

# 최고 점수 찾기 및 차트에 표시
max_score = max(oob_scores)
max_idx = n_estimators_range[oob_scores.index(max_score)]
ax.annotate(f'⭐ 최고 점수: {max_score:.4f}\n(트리 {max_idx}개)',
            xy=(max_idx, max_score),
            xytext=(max_idx + 5, max_score - 0.005),
            arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1.5, headwidth=8),
            fontsize=12, color='#d62728', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", lw=1.5, alpha=0.9))

# Streamlit에 차트 출력
st.pyplot(fig)

st.success("""
**📊 시각화 인사이트:**
1. **트리의 개수가 적을 때:** OOB 점수의 변동이 심하고 불안정합니다. (숲이 충분히 조성되지 않음)
2. **트리의 개수가 늘어날수록:** OOB 점수가 점점 상승하다가 일정한 수준(약 96% 대)에서 **안정적으로 수렴**하는 것을 볼 수 있습니다.
3. **효율성:** 교차 검증(Cross Validation)처럼 데이터를 분할하고 여러 번 학습할 필요 없이, **단 한 번의 학습만으로 자체적인 검증 스코어(OOB Score)를 얻을 수 있다는 점**이 매우 강력한 장점입니다!
""")

st.divider()

st.markdown("""
### ✨ 마무리 요약
- `RandomForestClassifier(oob_score=True)` 옵션 하나면 검증 세트 없이 모델의 예측력을 평가할 수 있습니다.
- 데이터가 적어서 훈련/검증/테스트 세트로 나누기 아까울 때 OOB 평가가 아주 유용하게 쓰입니다.
""")
