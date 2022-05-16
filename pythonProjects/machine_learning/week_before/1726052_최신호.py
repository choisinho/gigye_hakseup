import pandas as pd

data = pd.read_csv('sample.csv')
# sample.csv 파일 불러오기-----------------------↑↑↑

r_sum = []
r_avg = []

for i in range(20):
    now_sum = data.loc[i, ['운영체제', '논리회로', 'DB']].values.sum()
    r_sum.append(now_sum)
for i in range(20):
    now_avg = int(data.loc[i, ['운영체제', '논리회로', 'DB']].values.mean())  # 소수점은 제거하였음
    r_avg.append(now_avg)

data.insert(len(data.columns), '총합', r_sum)
data.insert(len(data.columns), '평균', r_avg)
# 각 학생의 총합, 평균 값 넣기-----------------------↑↑↑

def rank_by_avg(avg):
    if avg >= 90:
        return 'A'
    elif avg >= 80:
        return 'B'
    else:
        return 'C'
# A, 80이상 B, 70이상 C 등급을 구분하는 함수 만들기-----------------------↑↑↑

data = data.sort_values(ascending=True, by='총합')
#총합을 기준으로 오름차순 정렬하기-----------------------↑↑↑

data['학점'] = data['평균'].apply(rank_by_avg)
# 학점 열에 등급 추가하기-----------------------↑↑↑

print(data)
#출력-----------------------↑↑↑
