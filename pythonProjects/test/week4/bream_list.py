
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

#위에 애들이 물고기 데이터

# import matplotlib.pyplot as plt
# #시각화해주는 라이브러리
#
# plt.scatter(bream_length, bream_weight) #scatter 그래프를 사용하겠다.
# plt.scatter(smelt_length, smelt_weight ) #scatter 그래프를 사용하겠다.
#
# #물고기 세트 하나 더 추가 (빙어버전)
#
# plt.xlabel('length') #x축 : 길이
# plt.ylabel('weight)') #y축 : 높이
# plt.show() #항상 해야함

t_length = bream_length + smelt_length
t_weight = bream_weight + smelt_weight
#도미, 빙어 길이, 무게 각각 합치기

fish_data = [[l, w] for l, w in zip(t_length, t_weight)]
#사이킷런을 사용할 때는 2차원 배열의 형태로 데이터를 넣어줘야 한다. (길이, 무게)

fish_target = [1]*35 + [0]*14
#정답을 준비한다. 인공지능이 학습할 수 있도록. 1은 도미, 0은 빙어다.

from sklearn.neighbors import KNeighborsClassifier
#from은 라이브러리의 일부만 임포트하는 키워드

kn = KNeighborsClassifier()

kn.fit(fish_data, fish_target)
#fish_data의 답은 fish_target이라고 학습시킴.

kn.score(fish_data, fish_target)
#학습한 데이터와 똑같이 넣으면 1.0이라는 결과가 나옴. 1.0은 일치한 정답률(100%)를 의미

kn49 = KNeighborsClassifier(n_neighbors=49) #최근접 이웃의 기본값은 5지만 49로 하면 전체가 된다.
kn.fit(fish_data, fish_target)
kn.score(fish_data, fish_target)
#출력결과는 0.7어쩌구 이렇게 나오는데 이거는 35(도미)/49의 결과로, 전체 중 도미의 비중이 나온다.

result = kn.predict([[30, 600]]) #이 값은 도미인가? 빙어인가?? (예측)
print(result)
#도미에 가깝다(확률 계산)
