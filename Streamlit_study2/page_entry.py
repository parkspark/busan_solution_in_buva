# uv run streamlit run page_entry.py

# 진입점 entry point
import streamlit as st

# page
home = st.Page("page/home.py", title='홈', icon='🏠')
dashboard = st.Page("page/dashboard.py", title='대시보드', icon='💰')

# Navigation
pg = st.navigation([home, dashboard])
pg.run()
