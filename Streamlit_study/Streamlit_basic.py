'''
**B1** 사이드바 필터 4종 · **B2** metric 3개  
**B3** tabs(차트/데이터) · **B4** 차트 종류 바꿔보기
'''
# BASIC — 타이타닉 필터 대시보드
import streamlit as st
import pandas as pd
import plotly.express as px

# streamlit run .\Streamlit_study\Streamlit_basic.py

@st.cache_data
def load_titanic():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    return pd.read_csv(url)

df = load_titanic().copy()
df['Age'] = df['Age'].fillna(df['Age'].median())



# B1 사이드바 필터 문제
with st.sidebar:
    st.header("b1. 필터 문제")
    pclass_options = st.multiselect("객실 등급", [1, 2, 3], default=[1, 2, 3])
    gender = st.selectbox("성별", ["전체", "male", "female"])
    age_range = st.slider("나이 범위", 0, 80, (0, 80))
    survived_only = st.checkbox("생존자만")
    Fare_range = st.slider("요금 범위", 0,515, (0,515))


filtered = df[df['Pclass'].isin(pclass_options)]
filtered = filtered[(filtered['Age'] >= age_range[0]) & (filtered['Age'] <= age_range[1])]
if gender != "전체": filtered = filtered[filtered['Sex'] == gender]
if survived_only:    filtered = filtered[filtered['Survived'] == 1]

# b2. 메트릭
st.title("b2. 메트릭")
col1, col2, col3 = st.columns(3)
with col1: st.metric("선택된 승객", len(filtered))
with col2: st.metric("생존자", filtered['Survived'].sum())
with col3:
    rate = f"{filtered['Survived'].mean()*100:.1f}%" if len(filtered) > 0 else "N/A"
    st.metric("생존율", rate)

# b3. tabs
tab1, tab2, tab3 = st.tabs(["📊 차트", "📋 데이터", "b4 차트"])
with tab1:
    fig = px.histogram(filtered, x='Age', nbins=20,
                       title='나이 분포', template='simple_white')
    st.plotly_chart(fig, width='stretch')


with tab2:
    st.dataframe(filtered[['Name','Survived','Pclass','Sex','Age','Fare']],
                 hide_index=True)
    
with tab3:
    fig = px.histogram(
        filtered, 
        x="Age", 
        color="Survived",          
        barmode="overlay",
        nbins=20,
        title="생존 여부별 나이 분포", 
        template="simple_white"
    )
    st.plotly_chart(fig, width='stretch')

'''
**B1** 사이드바 필터 4종 · **B2** metric 3개  
**B3** tabs(차트/데이터) · **B4** 차트 종류 바꿔보기
'''


