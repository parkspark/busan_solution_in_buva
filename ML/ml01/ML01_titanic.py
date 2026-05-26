import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_titanic():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    return pd.read_csv(url)
titanic = load_titanic().copy()
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())



st.markdown("""
타이타닉 데이터를 탐색하고 예측 대상·ML 유형·특성 가설 3개를 구술 또는 메모로 정의할 수 있다

1. 데이터 로드 후 컬럼 목록 확인
2. Survived 분포 value_counts()
3. 결측값 컬럼 확인 isnull().sum()
""")


st.divider()
st.write("## " +"1. 데이터 로드 후 컬럼 목록 확인")

st.write("### " +"타이타닉 데이터")
st.dataframe(titanic.head(), use_container_width=True)

st.write("#### " + "titaninc 의 columns 목록")
st.write(list(titanic.columns))

st.divider()
st.write("## " +"2. Survived 분포 value_counts()")
st.dataframe(titanic['Survived'].value_counts())
st.write("#### " + "survived 컬럼에 대한 기술통계값 확인")
st.dataframe(titanic['Survived'].describe())

st.divider()
st.write("## " +"3.결측값 컬럼 확인")
st.caption("isnull().sum()")
st.dataframe(titanic.isnull().sum())

st.divider()

st.write("## " + "4. 머신러닝 문제 정의 & 특성 가설 수립")

# 1. 머신러닝 문제 정의
st.markdown("""타이타닉 데이터셋을 활용해 해결할 수 있는 가장 대표적인 머신러닝 문제입니다.

1. **지도학습** (Supervised Learning) - **이진 분류** (Binary Classification)
   * **예측 대상** (Label/Target): `Survived` (생존 여부: 1=생존, 0=사망)
   * **설명**: 승객들의 인적 정보와 탑승 정보를 Feature로 받아 이 사람이 살았을지 죽었을지를 `1` 또는 `0` 으로 예측합니다.
   
2. **지도학습** (Supervised Learning) - **회귀** (Regression)
   * **예측 대상** (Label/Target): `Fare` (탑승 요금)
   * **설명**: 객실 등급(`Pclass`), 탑승 항구(`Embarked`), 동반 가족 수 등의 Feature를 기반으로 적정 운임 요금(수치형 값)을 예측합니다.""")

st.write("### 머신러닝 예측을 위한 특성 가설 (Feature Hypotheses)")
st.markdown("데이터를 분석하고 모델을 학습시키기 전, 도메인 지식과 상식에 기반해 세울 수 있는 유의미한 가설들입니다.")

# 가설 1
st.info("""**가설 1. 성별 및 연령 가설** ("여성과 아이 우선")
* **내용**: 재난 상황에서 여성(`Sex`)과 어린이(`Age`)가 우선적으로 구명보트에 탑승했을 것이므로, **여성이고 나이가 어릴수록 생존율이 더 높을 것이다.**
* **관련 특성** (Features): `Sex` (성별), `Age` (나이)""")

# 가설 2
st.success("""**가설 2. 사회경제적 등급 가설** ("객실 위치와 부")
* **내용**: 1등석 승객들은 구명보트와 가까운 상부 갑판에 객실이 배치되었고, 높은 운임(`Fare`)을 지불했으므로 **객실 등급(`Pclass`)이 높고 운임이 비쌀수록 생존율이 높을 것이다.**
* **관련 특성** (Features): `Pclass` (티켓 등급), `Fare` (운임 요금)""")

# 가설 3
st.warning("""**가설 3. 동반 가족 수 가설** ("가족은 시너지인가, 족쇄인가?")
* **내용**: 홀로 탑승한 승객(1인)보다 동반 가족이 있는 승객이 서로 도와 생존에 유리할 수 있으나, 가족 규모가 너무 큰 대가족은 이동 및 의사결정이 늦어져 **가족 수가 적당할 때 (2~4명) 생존율이 가장 높고, 1인이거나 대가족 (5인 이상)일 때는 생존율이 낮을 것이다.**
* **관련 특성** (Features): `Sib/Sp` (형제자매/배우자 수), `Par/ch` (부모/자녀 수) *두 특성을 더해 '가족 크기'라는 새로운 특성 생성 가능*""")

# 가설 4
st.error("""**가설 4. 탑승 항구 가설** ("배후 지역의 부유함")
* **내용**: 부유한 도시인 쉘부르(`C` = Cherbourg)에서 탑승한 승객들이 비교적 1등석 비율이 높을 것이므로, **탑승 항구(`Embarked`)가 C인 승객들의 생존율이 상대적으로 높을 것이다.**
* **관련 특성** (Features): `Embarked` (탑승 항구), `Pclass` (티켓 등급)""")
