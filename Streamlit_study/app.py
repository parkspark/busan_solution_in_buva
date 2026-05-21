import streamlit as st
import pandas as pd
import plotly.express as px

# streamlit run .\Streamlit_study\app_by_gemini.py

st.title("Streamlit Study")
st.header("위젯")
st.text("위젯? 사용자 입력을 받는 UI 요소들")

# Text input
name = st.text_input("이름 : ", placeholder="이름입력칸")
st.write(f"{name}님 반강ㅂ다")

# dropdown select
gender = st.selectbox("성별", ["선택", 'male', 'femael'])
st.write("성별 : ", gender)

#button
if st.button("확인"):
    st.write("버튼이 눌렸습니다!")
# 체크박스
show = st.checkbox("원본 데이터 보기")
# 다중 선택
opts = st.multiselect("등급", [1, 2, 3], default=[1, 2, 3])


# with 블록 안 = 사이드바
with st.sidebar:
    st.header("필터")
    pclass = st.selectbox("객실 등급", [1, 2, 3])
    survived_only = st.checkbox("생존자만")
# 메인 화면
st.title("타이타닉 대시보드")
st.write(f"선택된 등급: {pclass}")

@st.cache_data
def load_titanic():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    return pd.read_csv(url)
titanic = load_titanic().copy()
# ── 위젯 ──────────────────────────────────────────────────────────────────────
pclass = st.selectbox("객실 등급", [1, 2, 3])
survived_only = st.checkbox("생존자만 보기")
# ── 필터 적용 ─────────────────────────────────────────────────────────────────
filtered = titanic[titanic['Pclass'] == pclass]
if survived_only:
    filtered = filtered[filtered['Survived'] == 1]
st.write(f"결과: {len(filtered)}명")
st.dataframe(filtered.head(20))


st.metric(label="생존율", value="38.4%", delta="2.2%")
st.metric(label="생존율", value="35.4%", delta="-2.2%")
st.metric(label="가격", value="1500", delta="-2.2"
          ,delta_color="inverse")
st.metric(label="가격", value="1500", delta="-2.2"
          ,delta_color="off")


# plotly

# imdb = pd.read_csv('./강의PDF/imdb_top_1000.csv')

# fig = px.scatter(
#     imdb,
#     x='No_of_Votes',
#     y='IMDB_Rating',
#     hover_name='Series_Title',      
#     # 굵게 표시
#     hover_data=['Released_Year',
#     'Genre'],
#     title='투표 수 vs IMDB 평점',
#     template='simple_white',
#     opacity=0.6
# )
# fig.show()
# fig.write_html('imdb_scatter.html')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("전체 승객", 891)
with col2:
    st.metric("생존자", 342)
with col3:
    st.metric("생존율", "38.4%")

@st.cache_data
def load_titanic():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    return pd.read_csv(url)
titanic = load_titanic()

fig = px.scatter(
    titanic, x='Age', y='Fare',
    color='Survived',
    hover_name='Name',
    title='나이 vs 요금',
    template='simple_white'
)
# ✅ width='stretch': 전체 너비 (최신 API)
st.plotly_chart(fig, width='stretch')

# 균등 3분할
col1, col2, col3 = st.columns(3)
# ⚠ 변수 수 = columns() 숫자 (반드시 일치)
with col1:
    st.metric("전체 승객", 891)
with col2:
    st.metric("생존자", 342)
with col3:
    st.metric("생존율", "38.4%")
# 비율 지정 (3:1)
col_wide, col_narrow = st.columns([3, 1])

tab1, tab2, tab3 = st.tabs(["📊 차트", "📋 데이터", "ℹ 정보"])
with tab1:
    st.plotly_chart(fig, width='stretch')
with tab2:
    st.dataframe(df, hide_index=True)
with tab3:
    st.write("데이터셋 설명")

