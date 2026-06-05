import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff

# 1. 데이터셋 불러오기 (.arff 파일 로드)
arff_path = r"c:\Users\금정산2-PC15\Desktop\busan_solution_in_buva\강의자료\DryBeanDataset\Dry_Bean_Dataset.arff"
data, meta = arff.loadarff(arff_path)
df = pd.DataFrame(data)

# 'Class' 컬럼의 바이트 문자열을 일반 문자열로 디코딩
df['Class'] = df['Class'].str.decode('utf-8')

# 피처(X)와 타겟(y) 분리
X = df.drop(columns=['Class'])
y = df[['Class']]

# 2. 기본 정보 확인
print("=== 데이터 정보 ===")
df.info()

# 3. 피처 통계치 분포 확인
print("\n=== 피처 분포 (수치형 변수 기술통계) ===")
print(X.describe().T)

# 4. 클래스 분포 확인 (타겟 변수)
print("\n=== 클래스(품종) 분포 ===")
class_counts = y['Class'].value_counts()
print(class_counts)

# 5. 주요 피처 시각화 (Box Plot - 면적 분포)
plt.figure(figsize=(12, 8))
sns.boxplot(x='Class', y='Area', data=df)
plt.title('Area Distribution by Bean Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. 피처 간 상관관계 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, annot_kws={"size": 8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
