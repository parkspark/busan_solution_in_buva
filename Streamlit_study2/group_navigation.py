# uv run streamlit run group_navigation.py

import streamlit as st

overview = st.Page("group_navigation/overview.py", title="요약",     icon="📊")
detail   = st.Page("group_navigation/detail.py",   title="상세 분석", icon="🔍")
setting = st.Page('group_navigation/setting.py', title='설정', icon='⚙️')

pg = st.navigation({"분석": [overview, detail], "설정": [setting]})
pg.run()