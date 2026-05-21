# streamlit run .\Streamlit_study\Streamlit_challenge.py
# 그냥 실행시켰을경우(vscode 기준, 
# Thread 'MainThread': missing ScriptRunContext! 
# This warning can be ignored when running in bare mode.) 에러 발생
# 따라서 임시방편으로 cmd에 위 명령어로 실행하여 확인한다.


import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_imdb():
    df = pd.read_csv('./강의자료/imdb_top_1000.csv')
    df['Released_Year'] = pd.to_numeric(
        df['Released_Year'], errors='coerce')
    return df

imdb = load_imdb().copy()

@st.cache_data
def load_titanic():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    return pd.read_csv(url)
titanic = load_titanic().copy()
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

# @st.cache_data
# def load_apt():
#     df = pd.read_csv('./강의자료/seoul_apartment.csv')
#     df['계약년월_str'] = df['계약년월'].astype(str)
#     df['구'] = df['시군구'].str.split().str[1]
#     return df
# sa = load_apt().copy()


with st.sidebar:
    st.header("IMDB 필터")
    min_rating = st.slider("최소 평점", 7.0, 9.5, 7.6, step=0.1)
    year_range = st.slider("개봉 연도", 1920, 2020, (2000, 2020))
    
    st.divider()
    st.divider()
    st.divider()

    st.caption("C3. titanic 데이터 사이드바")
    pclass_options = st.multiselect("객실 등급", [1, 2, 3], default=[1, 2, 3])
    gender = st.selectbox("성별", ["전체", "male", "female"])
    age_range = st.slider("나이 범위", 0, 80, (0, 80))
    survived_only = st.checkbox("생존자만")
    # Fare_range = st.slider("요금 범위", 0,515, (0,515))


filtered_imdb = imdb[
    (imdb['IMDB_Rating'] >= min_rating) &
    (imdb['Released_Year'] >= year_range[0]) &
    (imdb['Released_Year'] <= year_range[1])
]

filtered_titaninc = titanic[titanic['Pclass'].isin(pclass_options)]
filtered_titaninc = filtered_titaninc[(filtered_titaninc['Age'] >= age_range[0]) & (filtered_titaninc['Age'] <= age_range[1])]
if gender != "전체": filtered_titaninc = filtered_titaninc[filtered_titaninc['Sex'] == gender]
if survived_only:    filtered_titaninc = filtered_titaninc[filtered_titaninc['Survived'] == 1]


st.title('Streamlit_chanllenge.py')
st.title('A1. Metric2 + 평점 히스토그램')

col1, col2 = st.columns(2)
with col1: 
    st.metric(label = "선택된 영화 개수", value = len(filtered_imdb))
with col2:
    avg_rating = filtered_imdb['IMDB_Rating'].mean() if len(filtered_imdb) > 0 else 0.0
    st.metric(label="선택된 영화의 평균 평점", value=f"{avg_rating:.2f}점")

# A2.
st.title("A2. Scatter 탭")

tab1, tab2 = st.tabs(["평점 Scatter", "영화 데이터"])

with tab1:
    st.subheader("A2. 문제 scatter")
    if len(filtered_imdb) > 0:
        fig_scatter = px.scatter(
            filtered_imdb,
            x="Released_Year",
            y="IMDB_Rating",
            color="IMDB_Rating", # 평점 높낮이에 따른 색상 변화
            hover_name="Series_Title",
            title="연도별 영화 평점 Scatter",
            labels={"Released_Year": "개봉 연도", "IMDB_Rating": "평점"},
            template="simple_white"
        )
        st.plotly_chart(fig_scatter, width='stretch')
    else:
        st.warning("선택한 필터 조건에 맞는 영화가 없습니다.")

with tab2:
    st.subheader("필터링된 영화 목록 (상위 50개)")
    display_cols = ['Series_Title', 'Released_Year', 'IMDB_Rating', 'Genre', 'Director']
    available_cols = [col for col in display_cols if col in filtered_imdb.columns]
    st.dataframe(filtered_imdb[available_cols].head(50), hide_index=True)



st.title("C1. st.tabs")
tab1, tab2, tab3 = st.tabs(['📊 나이 분포','📊 등급별 생존','📋 데이터'])

with tab1:
    st.caption("📊 나이 분포")
    fig = px.histogram(filtered_titaninc, x='Age', nbins=20,
                        title='나이 분포', template='simple_white')
    st.plotly_chart(fig, width='stretch')

with tab2:
    st.caption("📊 등급별 생존")
    pclass_survived = filtered_titaninc.groupby('Pclass')['Survived'].sum().reset_index()
    pclass_survived['Pclass_Label'] = pclass_survived['Pclass'].astype(str) + "등급"
    fig_pie = px.pie(
        pclass_survived, 
        values='Survived',
        names='Pclass_Label',
        title='등급(Pclass)별 생존자 분포',
        template='simple_white',
        hole=0.3
    )
    fig_pie.update_traces(textinfo='percent+label', textposition='inside')
    st.plotly_chart(fig_pie, width='stretch')



with tab3:
    st.caption("📋 데이터")
    st.dataframe(filtered_titaninc[['Name','Survived','Pclass','Sex','Age','Fare']],
                 hide_index=True)

