import pandas as pd

data = pd.read_csv('행정구역_시군구_별__성별_인구수_20220427134504.csv')[10:18]
data = data.reset_index(drop=True)
print(data)

import matplotlib.pyplot as plt

ax = plt.gca()
coloumns = ['1992', '1997', '2002', '2007', '2012', '2017']
data.plot(kind="line", x='행정구역(시군구)별', y=coloumns, ax=ax)
plt.rcParams.update({'font.family': 'malgun gothic', 'font.size': 12})
plt.title("년도별 인구수")
plt.xlabel("년도")
plt.ylabel("인구수")
plt.show()