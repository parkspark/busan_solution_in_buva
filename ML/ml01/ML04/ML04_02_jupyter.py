import streamlit as st
import streamlit.components.v1 as components
import os

st.title("🍷 와인 데이터 결정 트리 & 앙상블 분석 (Jupyter Notebook)")

notebook_path = r"c:\Users\금정산2-PC15\Desktop\busan_solution_in_buva\ML\ml02\ml04ex-wine.ipynb"

if not os.path.exists(notebook_path):
    st.error(f"지정된 경로에서 노트북 파일을 찾을 수 없습니다: {notebook_path}")
else:
    try:
        import nbformat
        from nbconvert import HTMLExporter

        # 노트북 읽기
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # HTML 변환기 설정
        html_exporter = HTMLExporter()
        html_exporter.exclude_input_prompt = True
        html_exporter.exclude_output_prompt = True
        html_exporter.theme = 'light'
        
        # 변환 실행
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        # HTML을 위한 커스텀 스타일 (테두리 및 배경)
        custom_html = f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
            {body}
        </div>
        """
        
        # Streamlit 화면에 HTML 렌더링
        components.html(custom_html, height=1200, scrolling=True)
        
    except ImportError:
        st.warning("Jupyter Notebook을 HTML로 렌더링하기 위해 패키지 설치가 진행 중이거나 필요합니다.")
        st.info("앱을 새로고침 하거나, 터미널에서 다음 명령어를 실행해 주세요: `pip install nbformat nbconvert`")
    except Exception as e:
        st.error(f"노트북 파일을 변환하는 중 오류가 발생했습니다: {e}")
