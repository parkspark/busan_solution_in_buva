import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Page Config ---
st.set_page_config(page_title="마법의 숲 탐정단", page_icon="🌳", layout="wide")

st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; color: #1f2937; }
    /* Dark mode support for text inside cards */
    @media (prefers-color-scheme: dark) {
        .card { background-color: #1f2937; color: #f9fafb; border: 1px solid #374151; }
    }
</style>
""", unsafe_allow_html=True)

# --- Functions ---
def calc_gini(series):
    if len(series) == 0: return 0
    counts = series.value_counts()
    probs = counts / len(series)
    return 1 - np.sum(probs**2)

def calc_weighted_gini(df, split_col, target_col="마법동물"):
    total_len = len(df)
    weighted_gini = 0
    for val in df[split_col].unique():
        subset = df[df[split_col] == val]
        gini = calc_gini(subset[target_col])
        weighted_gini += (len(subset) / total_len) * gini
    return weighted_gini

class SimpleTree:
    """순수 파이썬으로 구현한 간단한 카테고리형 결정 트리 모델"""
    def __init__(self, df, features, target='마법동물'):
        self.tree = self.build_tree(df, features, target)
        
    def build_tree(self, df, features, target):
        # 1. 모든 데이터가 같은 클래스면 종료
        if len(df[target].unique()) == 1:
            return df[target].iloc[0]
        # 2. 더 이상 나눌 특징이 없으면 다수결로 종료
        if len(features) == 0:
            return df[target].mode()[0]
            
        best_f, best_gini = None, 2.0
        for f in features:
            g = calc_weighted_gini(df, f, target)
            if g < best_gini:
                best_gini = g
                best_f = f
                
        if best_f is None:
            return df[target].mode()[0]
            
        node = {'feature': best_f, 'branches': {}, 'default': df[target].mode()[0]}
        remaining_features = [f for f in features if f != best_f]
        
        for val in df[best_f].unique():
            subset = df[df[best_f] == val]
            node['branches'][val] = self.build_tree(subset, remaining_features, target)
        return node
        
    def predict_one(self, sample, node=None):
        if node is None: node = self.tree
        if not isinstance(node, dict): return node
        
        f = node['feature']
        val = sample[f]
        if val in node['branches']:
            return self.predict_one(sample, node['branches'][val])
        else:
            # 학습 데이터에서 보지 못한 범주가 나오면 default 반환
            return node['default']

# --- Data Initialization ---
CSV_PATH = os.path.join(os.path.dirname(__file__), 'animal_cards.csv')

if 'initialized' not in st.session_state or len(st.session_state.get('train_df', [])) != 20:
    # 가상의 동물 데이터 생성 (총 40마리)
    data = [
        {"이름": "사자", "다리": 4, "털": "O", "날개": "X", "서식지": "땅", "마법동물": "X", "이모지": "🦁"},
        {"이름": "독수리", "다리": 2, "털": "X", "날개": "O", "서식지": "하늘", "마법동물": "X", "이모지": "🦅"},
        {"이름": "상어", "다리": 0, "털": "X", "날개": "X", "서식지": "물", "마법동물": "X", "이모지": "🦈"},
        {"이름": "페가수스", "다리": 4, "털": "O", "날개": "O", "서식지": "하늘", "마법동물": "O", "이모지": "🦄"}, 
        {"이름": "인어", "다리": 0, "털": "O", "날개": "X", "서식지": "물", "마법동물": "O", "이모지": "🧜‍♀️"},
        {"이름": "구미호", "다리": 4, "털": "O", "날개": "X", "서식지": "숲", "마법동물": "O", "이모지": "🦊"},
        {"이름": "거미", "다리": 8, "털": "X", "날개": "X", "서식지": "땅", "마법동물": "X", "이모지": "🕷️"},
        {"이름": "박쥐", "다리": 2, "털": "O", "날개": "O", "서식지": "숲", "마법동물": "X", "이모지": "🦇"},
        {"이름": "용", "다리": 4, "털": "X", "날개": "O", "서식지": "하늘", "마법동물": "O", "이모지": "🐉"},
        {"이름": "개구리", "다리": 4, "털": "X", "날개": "X", "서식지": "물", "마법동물": "X", "이모지": "🐸"},
        {"이름": "슬라임", "다리": 0, "털": "X", "날개": "X", "서식지": "숲", "마법동물": "O", "이모지": "💧"},
        {"이름": "늑대인간", "다리": 2, "털": "O", "날개": "X", "서식지": "숲", "마법동물": "O", "이모지": "🐺"},
        {"이름": "고래", "다리": 0, "털": "X", "날개": "X", "서식지": "물", "마법동물": "X", "이모지": "🐳"},
        {"이름": "타조", "다리": 2, "털": "O", "날개": "X", "서식지": "땅", "마법동물": "X", "이모지": "🦤"},
        {"이름": "불사조", "다리": 2, "털": "O", "날개": "O", "서식지": "하늘", "마법동물": "O", "이모지": "🔥"},
        {"이름": "뱀", "다리": 0, "털": "X", "날개": "X", "서식지": "숲", "마법동물": "X", "이모지": "🐍"},
        {"이름": "그리폰", "다리": 4, "털": "O", "날개": "O", "서식지": "산", "마법동물": "O", "이모지": "🦅"}, 
        {"이름": "나비", "다리": 6, "털": "X", "날개": "O", "서식지": "숲", "마법동물": "X", "이모지": "🦋"},
        {"이름": "유니콘", "다리": 4, "털": "O", "날개": "X", "서식지": "숲", "마법동물": "O", "이모지": "🦄"},
        {"이름": "펭귄", "다리": 2, "털": "O", "날개": "X", "서식지": "물", "마법동물": "X", "이모지": "🐧"},
        {"이름": "기린", "다리": 4, "털": "O", "날개": "X", "서식지": "땅", "마법동물": "X", "이모지": "🦒"},
        {"이름": "코끼리", "다리": 4, "털": "X", "날개": "X", "서식지": "땅", "마법동물": "X", "이모지": "🐘"},
        {"이름": "부엉이", "다리": 2, "털": "O", "날개": "O", "서식지": "숲", "마법동물": "X", "이모지": "🦉"},
        {"이름": "켈피", "다리": 4, "털": "O", "날개": "X", "서식지": "물", "마법동물": "O", "이모지": "🐴"},
        {"이름": "해태", "다리": 4, "털": "O", "날개": "X", "서식지": "숲", "마법동물": "O", "이모지": "🦁"},
        {"이름": "가고일", "다리": 4, "털": "X", "날개": "O", "서식지": "산", "마법동물": "O", "이모지": "🗿"},
        {"이름": "원숭이", "다리": 2, "털": "O", "날개": "X", "서식지": "숲", "마법동물": "X", "이모지": "🐒"},
        {"이름": "악어", "다리": 4, "털": "X", "날개": "X", "서식지": "물", "마법동물": "X", "이모지": "🐊"},
        {"이름": "거북이", "다리": 4, "털": "X", "날개": "X", "서식지": "물", "마법동물": "X", "이모지": "🐢"},
        {"이름": "현무", "다리": 4, "털": "X", "날개": "X", "서식지": "물", "마법동물": "O", "이모지": "🐢"},
        {"이름": "백호", "다리": 4, "털": "O", "날개": "X", "서식지": "산", "마법동물": "O", "이모지": "🐯"},
        {"이름": "주작", "다리": 2, "털": "O", "날개": "O", "서식지": "하늘", "마법동물": "O", "이모지": "🦚"},
        {"이름": "청룡", "다리": 4, "털": "X", "날개": "X", "서식지": "하늘", "마법동물": "O", "이모지": "🐉"},
        {"이름": "불곰", "다리": 4, "털": "O", "날개": "X", "서식지": "숲", "마법동물": "X", "이모지": "🐻"},
        {"이름": "고양이", "다리": 4, "털": "O", "날개": "X", "서식지": "땅", "마법동물": "X", "이모지": "🐱"},
        {"이름": "스핑크스", "다리": 4, "털": "O", "날개": "O", "서식지": "사막", "마법동물": "O", "이모지": "🐈"},
        {"이름": "낙타", "다리": 4, "털": "O", "날개": "X", "서식지": "사막", "마법동물": "X", "이모지": "🐫"},
        {"이름": "맨티코어", "다리": 4, "털": "O", "날개": "O", "서식지": "산", "마법동물": "O", "이모지": "🦂"},
        {"이름": "전갈", "다리": 8, "털": "X", "날개": "X", "서식지": "사막", "마법동물": "X", "이모지": "🦂"},
        {"이름": "오리", "다리": 2, "털": "O", "날개": "O", "서식지": "물", "마법동물": "X", "이모지": "🦆"}
    ]
    df = pd.DataFrame(data)
    # CSV 파일로 저장
    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    
    # 데이터를 무작위로 섞고 훈련/테스트로 20마리씩 분리
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    st.session_state['train_df'] = df.iloc[:20].copy()
    st.session_state['test_df'] = df.iloc[20:].copy()
    st.session_state['initialized'] = True

train_df = st.session_state['train_df']
test_df = st.session_state['test_df']

# --- UI Header ---
st.title("🌳 마법의 숲 탐정단 (Random Forest Detectives) 🕵️‍♂️")
st.markdown("동물 카드를 통해 **결정 트리(Decision Tree)**와 **랜덤 포레스트(Random Forest)**의 원리를 쉽고 재미있게 알아보세요!")

tabs = st.tabs(["🍃 Stage 1: 결정 트리 기초", "🧩 Stage 2: 과적합 체험", "🌲 Stage 3: 탐정단 결성", "📊 Stage 4: OOB 기습 퀴즈"])

# --- Stage 1 ---
with tabs[0]:
    st.header("Stage 1: 하나의 질문으로 분류하기")
    st.write("20마리의 동물들을 하나의 특징으로 두 그룹으로 나누어 봅시다. 어떤 질문을 해야 '마법동물'과 '일반동물'을 가장 잘 구분할 수 있을까요?")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(train_df[['이모지', '이름', '다리', '털', '날개', '서식지', '마법동물']], hide_index=True, use_container_width=True)
    
    with col2:
        root_gini = calc_gini(train_df['마법동물'])
        st.markdown(f"**현재 불순도 (Gini):** `{root_gini:.3f}`")
        st.progress(float(root_gini * 2))  # 불순도는 최대 0.5이므로 시각화를 위해 2배
        st.caption("불순도가 0에 가까울수록 한 그룹에 같은 종류(마법/일반)만 모여있다는 뜻이에요!")
        
        split_feature = st.selectbox("어떤 특징으로 그룹을 나눌까요?", ["선택하세요", "다리", "털", "날개", "서식지"], key='s1_feature')
        
    if split_feature != "선택하세요":
        wgini = calc_weighted_gini(train_df, split_feature)
        st.success(f"**'{split_feature}'(으)로 나누었을 때의 평균 불순도:** `{wgini:.3f}`")
        st.progress(float(wgini * 2))
        
        groups = train_df.groupby(split_feature)
        cols = st.columns(len(groups))
        for col, (val, group) in zip(cols, groups):
            with col:
                g_gini = calc_gini(group['마법동물'])
                st.markdown(f"<div class='card'><h4>{split_feature} = {val}</h4><p>불순도: {g_gini:.3f}</p></div>", unsafe_allow_html=True)
                for _, row in group.iterrows():
                    color = "#9b59b6" if row['마법동물']=="O" else "#7f8c8d"
                    st.markdown(f"<span style='color:{color}; font-weight:bold;'>{row['이모지']} {row['이름']} (마법:{row['마법동물']})</span>", unsafe_allow_html=True)


    st.markdown("**분류의 정확도:** 불순도가 낮다는 것은 데이터가 섞여 있지 않고 특정 클래스(범주)로 깔끔하게 나뉘었음을 뜻합니다.")
    st.markdown("**의사결정나무의 목표:** 모델은 데이터를 분할할 때 불순도를 낮추는 방향으로 가지를 뻗어나가며, 최종적으로 불순도가 0이 되는 것을 목표로 합니다.")

# --- Stage 2 ---
with tabs[1]:
    st.header("Stage 2: 꼬치꼬치 캐묻기 (과적합, Overfitting)")
    st.write("질문을 너무 많이 해서 모든 동물을 1마리씩 완벽하게 쪼개면 어떻게 될까요? 특징들을 순서대로 선택해보세요!")
    
    selected_features = st.multiselect("특징들을 순서대로 여러 개 선택하세요:", ["다리", "털", "날개", "서식지"], key='s2_features')
    
    if selected_features:
        groups = train_df.groupby(selected_features)
        st.info(f"훈련 데이터가 총 **{len(groups)}개의 그룹**으로 잘게 쪼개졌습니다!")
        
        # 모델 예측 함수 (과적합 시뮬레이션용)
        def predict_overfit(sample):
            match_group = train_df.copy()
            for f in selected_features:
                match_group = match_group[match_group[f] == sample[f]]
            if len(match_group) > 0:
                return match_group['마법동물'].mode()[0]
            else:
                return "Unknown"
                
        # 훈련 데이터 정답률 계산
        train_correct = sum(1 for _, row in train_df.iterrows() if predict_overfit(row) == row['마법동물'])
        train_acc = (train_correct / len(train_df)) * 100
        
        # 테스트 데이터 정답률 계산 및 실패 사례 수집
        test_correct = 0
        failed_examples = []
        for _, row in test_df.iterrows():
            pred = predict_overfit(row)
            if pred == row['마법동물']:
                test_correct += 1
            else:
                reason = "데이터 없음" if pred == "Unknown" else pred
                failed_examples.append((row, reason))
                
        test_acc = (test_correct / len(test_df)) * 100
        
        col1, col2 = st.columns(2)
        col1.metric("훈련 데이터 정답률 (공부한 내용)", f"{train_acc:.1f}%")
        col2.metric("새로운 테스트 정답률 (실전 시험)", f"{test_acc:.1f}%")
        
        if train_acc > test_acc + 10 and len(selected_features) >= 2:
            st.warning("⚠️ **과적합(Overfitting) 발생!** 훈련 데이터의 정답은 잘 맞히지만, 너무 지엽적인 특징까지 외워버린 탓에 새로운 데이터에는 제대로 대처하지 못하고 있습니다.")
        elif train_acc == 100:
            st.success("훈련 데이터를 완벽하게 외웠습니다! 하지만 실전 테스트 점수는 어떨까요?")
            
        if failed_examples:
            st.divider()
            st.subheader("🚨 과적합 실패 사례 탐구")
            st.write("모델이 실전에서 어떻게 실수했는지 대표적인 사례 1개를 살펴봅시다.")
            
            # 첫 번째 실패 사례 가져오기
            sample_test, fail_reason = failed_examples[0]
            st.markdown(f"<div class='big-font'>{sample_test['이모지']} {sample_test['이름']}</div>", unsafe_allow_html=True)
            st.write(f"**실제 정답:** 마법동물 {sample_test['마법동물']}")
            st.write(f"**특징:** " + ", ".join([f"{f}={sample_test[f]}" for f in selected_features]))
            
            if fail_reason == "데이터 없음":
                st.error("💥 예측 불가! 훈련 데이터에 이런 특징 조합을 가진 동물이 하나도 없어서 모델이 당황했습니다. 너무 세세하게 기준을 나눈 탓입니다.")
            else:
                match_group = train_df.copy()
                for f in selected_features:
                    match_group = match_group[match_group[f] == sample_test[f]]
                st.write(f"👉 훈련 데이터에서 똑같은 특징을 가졌던 동물: {', '.join(match_group['이름'].tolist())} (마법동물 {fail_reason})")
                st.error(f"예측 실패! 모델은 예측 결과로 **마법동물 {fail_reason}**을(를) 내놓았습니다. 훈련 데이터에 있던 동물의 정답만 무작정 따라 하다 보니 일반화에 실패했습니다.")


# --- Stage 3 ---
with tabs[2]:
    st.header("Stage 3: 마법의 숲 탐정단 결성 (랜덤 포레스트)")
    st.write("하나의 트리가 과적합에 빠지기 쉽다면, 여러 개의 트리를 만들어서 다수결로 정하면 어떨까요? 이것이 바로 앙상블 기법인 '랜덤 포레스트'입니다.")
    
    num_trees = st.slider("결성할 탐정(트리)의 수를 선택하세요:", min_value=1, max_value=10, value=5)
    
    if st.button(f"🕵️‍♂️ 탐정단(트리 {num_trees}개) 결성하기!"):
        forest = []
        features_list = ["다리", "털", "날개", "서식지"]
        
        for i in range(num_trees):
            # 복원 추출을 이용한 배깅(Bagging) 시각화 (훈련 데이터 개수만큼 샘플링)
            bag = train_df.sample(n=20, replace=True)
            tree = SimpleTree(bag, features_list)
            st.session_state[f'tree_{i}'] = tree
            st.session_state[f'bag_{i}'] = bag
            
        st.session_state['num_trees'] = num_trees
        st.session_state['forest_built'] = True
        
    if st.session_state.get('forest_built', False):
        built_num = st.session_state.get('num_trees', 5)
        st.success("탐정단 결성 완료! 각 탐정은 훈련 데이터 전체가 아닌 **무작위 복원 추출(Bootstrap)**된 데이터만 보았기 때문에, 서로 다른 편향을 가진 전문가가 됩니다. (다양성 확보)")
        
        # 5개씩 줄바꿈하여 카드 렌더링
        for row_idx in range(0, built_num, 5):
            cols = st.columns(5)
            for i in range(5):
                tree_idx = row_idx + i
                if tree_idx < built_num:
                    bag = st.session_state[f'bag_{tree_idx}']
                    unique_count = len(bag['이름'].unique())
                    with cols[i]:
                        st.markdown(f"<div class='card'><b>탐정 {tree_idx+1}호</b><br/><span style='font-size:12px;'>학습한 동물: {unique_count}/20종</span></div>", unsafe_allow_html=True)
                        st.write("".join(bag['이모지'].tolist()[:5]) + "...")
                
        st.divider()
        st.subheader("새로운 사건 의뢰하기!")
        st.write(f"새로운 동물이 나타났습니다! {built_num}명의 탐정은 각자 배운 지식을 바탕으로 다르게 예측할 수 있습니다. 최종 결과는 **다수결(Majority Voting)**로 정해집니다.")
        
        test_options = [f"{row['이모지']} {row['이름']}" for _, row in test_df.iterrows()]
        selected_idx = st.selectbox("어떤 동물을 의뢰할까요?", range(len(test_options)), format_func=lambda x: test_options[x])
        
        sample = test_df.iloc[selected_idx]
        st.markdown(f"**의뢰 대상:** {sample['이모지']} {sample['이름']} (실제 정답: 마법동물 {sample['마법동물']})")
        
        preds = []
        # 예측 결과 5개씩 줄바꿈하여 렌더링
        for row_idx in range(0, built_num, 5):
            cols = st.columns(5)
            for i in range(5):
                tree_idx = row_idx + i
                if tree_idx < built_num:
                    tree = st.session_state[f'tree_{tree_idx}']
                    pred = tree.predict_one(sample)
                    preds.append(pred)
                    with cols[i]:
                        st.markdown(f"<div style='text-align:center; padding:10px; margin-bottom:10px; border-radius:5px; background-color:#e2e8f0; color:#1e293b;'>탐정{tree_idx+1}<br/><b>{pred}</b></div>", unsafe_allow_html=True)
                
        o_count = preds.count('O')
        x_count = preds.count('X')
        final_pred = 'O' if o_count > x_count else 'X'
        
        st.markdown(f"### 📢 다수결 최종 결과: **{final_pred}** (O: {o_count}표, X: {x_count}표)")
        if final_pred == sample['마법동물']:
            st.success("🎉 탐정단이 정답을 맞췄습니다! 개별 탐정은 틀릴 수 있지만, 여러 명의 의견을 모으면 훨씬 똑똑해집니다 (집단 지성).")
        else:
            st.warning("앗, 탐정단이 최종적으로 틀렸네요. 하지만 하나의 트리(탐정 1명)가 단독으로 판단하는 것보다는 평균적으로 오답률이 훨씬 낮습니다.")

# --- Stage 4 ---
with tabs[3]:
    st.header("Stage 4: OOB (Out-of-Bag) 평가")
    st.markdown("""
    **OOB(Out-Of-Bag) 데이터란?**  
    탐정들이 각자 훈련 데이터를 무작위로 뽑아(복원 추출) 공부할 때, **어떤 동물은 운이 없게도 단 한 번도 뽑히지 않고 가방(Bag) 밖에 남게 됩니다.**  
    랜덤 포레스트는 이렇게 '가방 밖에 남겨진 데이터(OOB)'를 마치 **실전 모의고사**처럼 활용하여 스스로의 실력을 평가합니다. 
    별도의 테스트 데이터를 만들 필요 없이 스스로 채점할 수 있다는 것이 엄청난 장점이죠!
    """)
    
    if not st.session_state.get('forest_built', False):
        st.warning("Stage 3에서 먼저 탐정단을 결성해주세요!")
    else:
        st.divider()
        st.subheader("🕵️‍♂️ 특정 동물의 OOB 평가 과정 지켜보기")
        built_num = st.session_state.get('num_trees', 5)
        
        oob_options = [f"{row['이모지']} {row['이름']} (정답: {row['마법동물']})" for _, row in train_df.iterrows()]
        oob_idx = st.selectbox("훈련 데이터 중 한 마리를 골라보세요:", range(len(oob_options)), format_func=lambda x: oob_options[x])
        
        sample_oob = train_df.iloc[oob_idx]
        
        # 이 동물을 한 번도 뽑지 못한(학습하지 못한) 탐정 찾기
        blind_detectives = []
        for i in range(built_num):
            bag = st.session_state[f'bag_{i}']
            if oob_idx not in bag.index:
                blind_detectives.append(i)
                
        if len(blind_detectives) == 0:
            st.info(f"앗! 모든 탐정({built_num}명)이 공부할 때 **{sample_oob['이름']}**을(를) 최소 한 번씩은 뽑아서 공부했네요. 이 동물은 OOB 평가에 사용할 수 없습니다. 다른 동물을 골라보세요!")
        else:
            st.write(f"👉 총 {built_num}명의 탐정 중, **탐정 {', '.join([str(idx+1)+'호' for idx in blind_detectives])}**는 공부할 때 **{sample_oob['이름']}**을(를) 한 번도 본 적이 없습니다!")
            st.write("이 동물에 대해 '본 적 없는 탐정들'만 모여서 다수결 예측(기습 퀴즈)을 진행합니다.")
            
            preds = []
            cols = st.columns(len(blind_detectives))
            for col, tree_idx in zip(cols, blind_detectives):
                tree = st.session_state[f'tree_{tree_idx}']
                pred = tree.predict_one(sample_oob)
                preds.append(pred)
                with col:
                    st.markdown(f"<div style='text-align:center; padding:10px; border-radius:5px; background-color:#e2e8f0; color:#1e293b;'>탐정{tree_idx+1}호<br/><b>{pred}</b></div>", unsafe_allow_html=True)
            
            o_count = preds.count('O')
            x_count = preds.count('X')
            final_pred = 'O' if o_count > x_count else 'X'
            
            st.markdown(f"#### 📢 OOB 기습 퀴즈 결과: **{final_pred}** (O: {o_count}표, X: {x_count}표)")
            if final_pred == sample_oob['마법동물']:
                st.success("정답입니다! 공부한 적 없는 데이터임에도 올바르게 예측했습니다. 일반화가 잘 되어있네요.")
            else:
                st.warning("오답입니다. 아직 일반화 성능이 완벽하지 않네요.")
                
        st.divider()
        st.subheader("📊 전체 훈련 데이터 OOB 점수 확인하기")
        st.write("위 과정을 OOB 조건이 성립하는 모든 훈련 데이터에 대해 반복한 뒤, 최종 평균 점수를 냅니다.")
        if st.button("전체 OOB 점수 계산"):
            correct = 0
            valid_oob_count = 0
            oob_results = []
            
            for idx, row in train_df.iterrows():
                voting_trees = []
                for i in range(built_num):
                    bag = st.session_state[f'bag_{i}']
                    if idx not in bag.index:
                        voting_trees.append(st.session_state[f'tree_{i}'])
                        
                if len(voting_trees) > 0:
                    valid_oob_count += 1
                    preds = [t.predict_one(row) for t in voting_trees]
                    o_count = preds.count('O')
                    x_count = preds.count('X')
                    final_pred = 'O' if o_count > x_count else 'X'
                    
                    is_correct = final_pred == row['마법동물']
                    if is_correct: correct += 1
                        
                    oob_results.append({
                        '이모지': row['이모지'],
                        '이름': row['이름'],
                        '모르는 탐정 수': f"{len(voting_trees)}명",
                        '예측': final_pred,
                        '정답': row['마법동물'],
                        '결과': "✅" if is_correct else "❌"
                    })
                    
            if valid_oob_count > 0:
                accuracy = (correct / valid_oob_count) * 100
                st.metric("최종 OOB 정확도 (Accuracy)", f"{accuracy:.1f}%")
                st.write(f"총 20개의 훈련 데이터 중 **{valid_oob_count}개**의 동물이 한 명 이상의 탐정에게 낯선(OOB) 데이터로 판정되어 평가에 사용되었습니다.")
                st.dataframe(pd.DataFrame(oob_results), hide_index=True, use_container_width=True)
            else:
                st.write("우연히 모든 데이터가 모든 트리에 최소 한 번씩 뽑혔습니다. (매우 드문 확률!) 탐정단을 다시 결성해보세요.")
