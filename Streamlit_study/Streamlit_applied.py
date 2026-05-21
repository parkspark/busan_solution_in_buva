#**A1** metric 2개 + 평점 히스토그램  
#**A2** scatter 탭 (hover_name='Series_Title')

# streamlit run .\Streamlit_study\Streamlit_applied.py

import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_imdb():
    # df = pd.read_csv('강의자료\imdb_top_1000.csv')
    df = pd.read_csv('./강의자료/imdb_top_1000.csv')
    df['Released_Year'] = pd.to_numeric(
        df['Released_Year'], errors='coerce')
    return df

imdb = load_imdb().copy()

with st.sidebar:
    st.header("IMDB 필터")
    min_rating = st.slider("최소 평점", 7.0, 9.5, 7.6, step=0.1)
    year_range = st.slider("개봉 연도", 1920, 2020, (2000, 2020))

filtered = imdb[
    (imdb['IMDB_Rating'] >= min_rating) &
    (imdb['Released_Year'] >= year_range[0]) &
    (imdb['Released_Year'] <= year_range[1])
]

st.title('A1. Metric2 + 평점 히스토그램')

col1, col2 = st.columns(2)
with col1: 
    st.metric(label = "선택된 영화 개수", value = len(filtered))
with col2:
    avg_rating = filtered['IMDB_Rating'].mean() if len(filtered) > 0 else 0.0
    st.metric(label="선택된 영화의 평균 평점", value=f"{avg_rating}점")

# A2.
st.title("A2. Scatter 탭")

tab1, tab2 = st.tabs(["평점 Scatter", "영화 데이터"])

with tab1:
    st.subheader("A2. 문제 scatter")
    if len(filtered) > 0:
        fig_scatter = px.scatter(
            filtered,
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
    available_cols = [col for col in display_cols if col in filtered.columns]
    st.dataframe(filtered[available_cols].head(50), hide_index=True)



        