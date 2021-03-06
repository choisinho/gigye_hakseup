from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

t_length = bream_length + smelt_length
t_weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(t_length, t_weight)]
fish_target = [1]*35 + [0]*14

# train_input = fish_data[:35]
# train_target = fish_target[:35]
#
# test_input = fish_data[35:]
# test_target = fish_target[35:]
# #
kn = KNeighborsClassifier()
# kn = kn.fit(train_input, train_target)
#
# print(kn.score(test_input, test_target))
#0,0 (0% 불일치)
#훈련세트와 테스트세트가 섞이지 않았기 때문(완전 다른애들끼리 해놨음)

import numpy as np

# input_arr = np.array(fish_data)
# target_arr = np.array(fish_target)
# #numpy로 섞어주기
#
# # print(input_arr)
#
# index = np.arange(49)
# # print(index)
# #리스트 만들기
#
# np.random.seed(42) #랜덤으로 하되, 시드가 똑같으면 랜덤 배치가 똑같음
# np.random.shuffle(index)
# # print(index)
# #셔플하면 랜덤으로 바꿔줌
#
# train_input = input_arr[index[:35]]
# train_target = target_arr[index[:35]]
#
# test_input = input_arr[index[35:]]
# test_target = target_arr[index[35:]]

# import matplotlib.pyplot as plt
#
# kn = kn.fit(train_input, train_target)
# print(kn.score(test_input, test_target))
#
# plt.scatter(train_input[:,0], train_input[:, 1])
# plt.scatter(test_input[:,0], test_input[:, 1])
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()
#
# #실행해보면 훈련데이터도 들어가있다
# #파란색은 타깃 주황색은 테스트

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
print(train_input, train_target)

kn = kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))
print(kn.predict([[25, 150]]))
#
# import matplotlib.pyplot as plt
#
# plt.scatter(train_input[:,0], train_input[:, 1])
# plt.scatter(test_input[:,0], test_input[:, 1])
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()