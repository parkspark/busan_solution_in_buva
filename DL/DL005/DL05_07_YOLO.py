import streamlit as st
import numpy as np
import cv2
from PIL import Image
import random

# 페이지 설정
st.set_page_config(page_title="객체 탐지 알고리즘 비교", layout="wide")

# 1. 상단 화면 (제목 및 설명)
st.title("객체 탐지(Object Detection) 알고리즘 작동 원리 비교")

st.markdown("""
- **Sliding Window**: 고정된 윈도우 창을 이미지 전역에 조금씩 움직이며 수천 번 CNN 연산을 적용합니다. (극도로 느림)
- **R-CNN 계열**: 물체가 있을 법한 영역을 먼저 후보군(Region Proposal)으로 뽑아낸 후, 별도의 CNN으로 정밀 계산하는 2단계(Two-Stage) 방식을 사용합니다. (실시간 연산 불가능)
- **YOLO (Single-Stage)**: 하나의 통합된 신경망 구조가 전체 이미지로부터 객체 바운딩 박스 좌표와 클래스 확률을 동시에 direct로 예측합니다. (실시간 연산 가능)
""")

# 2. 입력부 (메인 화면)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

def process_sliding_window(img):
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    win_w, win_h = w // 5, h // 5
    step = min(w, h) // 10
    
    overlay = img_copy.copy()
    for y in range(0, h - win_h + 1, step):
        for x in range(0, w - win_w + 1, step):
            # RGB에서 (255, 0, 0)은 붉은색
            cv2.rectangle(overlay, (x, y), (x + win_w, y + win_h), (255, 0, 0), 2)
    
    # 투명도 적용
    cv2.addWeighted(overlay, 0.4, img_copy, 0.6, 0, img_copy)
    return img_copy

def process_rcnn(img):
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    overlay = img_copy.copy()
    # 30~50개의 무작위 노란색 후보 영역(Region Proposals)
    num_boxes = random.randint(30, 50)
    for _ in range(num_boxes):
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        bw = random.randint(50, w // 2)
        bh = random.randint(50, h // 2)
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, cx + bw // 2)
        y2 = min(h, cy + bh // 2)
        # RGB에서 (255, 255, 0)은 노란색
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
    cv2.addWeighted(overlay, 0.5, img_copy, 0.5, 0, img_copy)
    
    # 1~2개의 굵은 초록색 최종 박스
    num_final = random.randint(1, 2)
    for _ in range(num_final):
        cx = random.randint(w // 3, 2 * w // 3)
        cy = random.randint(h // 3, 2 * h // 3)
        bw = random.randint(100, w // 2)
        bh = random.randint(100, h // 2)
        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, cx + bw // 2)
        y2 = min(h, cy + bh // 2)
        # RGB에서 (0, 255, 0)은 초록색
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
    return img_copy

def process_yolo(img):
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    grid_size = 7
    cell_w = w / grid_size
    cell_h = h / grid_size
    
    overlay = img_copy.copy()
    
    # 그리드 격자선 그리기
    for i in range(1, grid_size):
        cv2.line(overlay, (int(i * cell_w), 0), (int(i * cell_w), h), (200, 200, 200), 1)
        cv2.line(overlay, (0, int(i * cell_h)), (w, int(i * cell_h)), (200, 200, 200), 1)
        
    cv2.addWeighted(overlay, 0.6, img_copy, 0.4, 0, img_copy)
    
    # 특정 1~2개의 그리드 셀 하이라이트 및 박스 그리기
    num_objects = random.randint(1, 2)
    for _ in range(num_objects):
        gx = random.randint(2, grid_size - 3)
        gy = random.randint(2, grid_size - 3)
        
        cx1 = int(gx * cell_w)
        cy1 = int(gy * cell_h)
        cx2 = int((gx + 1) * cell_w)
        cy2 = int((gy + 1) * cell_h)
        
        # 그리드 셀 하이라이트 (파란색)
        cell_overlay = img_copy.copy()
        # RGB에서 (0, 0, 255)는 파란색
        cv2.rectangle(cell_overlay, (cx1, cy1), (cx2, cy2), (0, 0, 255), -1)
        cv2.addWeighted(cell_overlay, 0.4, img_copy, 0.6, 0, img_copy)
        
        # 바운딩 박스 그리기
        bw = random.randint(int(cell_w * 1.5), int(cell_w * 3))
        bh = random.randint(int(cell_h * 1.5), int(cell_h * 3))
        cx = cx1 + cell_w / 2
        cy = cy1 + cell_h / 2
        
        x1 = int(max(0, cx - bw / 2))
        y1 = int(max(0, cy - bh / 2))
        x2 = int(min(w, cx + bw / 2))
        y2 = int(min(h, cy + bh / 2))
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # 텍스트와 배경 추가
        prob = random.randint(90, 99)
        text = f"Object {prob}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(img_copy, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 255), -1)
        cv2.putText(img_copy, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
        
    return img_copy

# 3 & 4. 조작 및 시각화 부 (메인 화면)
if uploaded_file is None:
    st.info("시각화할 이미지를 업로드해주세요.")
else:
    # 이미지를 RGB로 읽어오기
    image = Image.open(uploaded_file).convert("RGB")
    img_arr = np.array(image)
    
    # 3개의 탭 생성
    tab1, tab2, tab3 = st.tabs(["🪟 Sliding Window", "🔍 R-CNN (Two-Stage)", "⚡ YOLO (Single-Stage)"])
    
    with tab1:
        st.image(process_sliding_window(img_arr), use_container_width=True)
        st.info("이미지 전체를 수많은 박스로 잘라서 모두 검사하는 무식하지만 확실한 방법입니다. 연산량이 기하급수적으로 많습니다.")
        
    with tab2:
        st.image(process_rcnn(img_arr), use_container_width=True)
        st.info("Selective Search 등을 통해 물체가 있을 만한 '후보 영역'을 먼저 찾아내고, 그 영역들만 CNN으로 검사하여 효율성을 높였습니다.")
        
    with tab3:
        st.image(process_yolo(img_arr), use_container_width=True)
        st.info("이미지를 그리드로 나누고, 한 번의 신경망 연산(Single-Shot)만으로 각 그리드에서 박스 위치와 클래스 확률을 동시에 예측하여 실시간 탐지가 가능해졌습니다.")
