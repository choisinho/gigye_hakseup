#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from matplotlib.cbook import flatten

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


train_dir = 'C:/Users/dkwhs/Downloads/train/train'
test_dir = 'C:/Users/dkwhs/Downloads/dogs_vs_cats_small/test2'
validation_dir = 'C:/Users/dkwhs/Downloads/dogs_vs_cats_small/vadlidation2'

train_dogs_dir = 'C:/Users/dkwhs/Downloads/train/train/dogs'
train_cats_dir = 'C:/Users/dkwhs/Downloads/train/train/cats'

test_dogs_dir = 'C:/Users/dkwhs/Downloads/dogs_vs_cats_small/test2/dogs'
test_cats_dir = 'C:/Users/dkwhs/Downloads/dogs_vs_cats_small/test2/cats'

validation_dogs_dir = 'C:/Users/dkwhs/Downloads/dogs_vs_cats_small/validation2/dogs'
validation_cats_dir = 'C:/Users/dkwhs/Downloads/dogs_vs_cats_small/validation2/cats'


# In[3]:


#listdir(): 해당 폴더에 있는 파일을 가져온다.
print("훈련 개 데이터 수 : {}".format(len(os.listdir(train_dogs_dir))))
print("훈련 고양이 데이터 수 : {}".format(len(os.listdir(train_cats_dir))))

print("테스트 개 데이터 수 : {}".format(len(os.listdir(test_dogs_dir))))
print("테스트 고양이 데이터 수 : {}".format(len(os.listdir(test_cats_dir))))

print("평가 개 데이터 수 : {}".format(len(os.listdir(validation_dogs_dir))))
print("평가 고양이 데이터 수 : {}".format(len(os.listdir(validation_cats_dir))))


# In[4]:


#스케일링
train_gen = ImageDataGenerator( rescale = 1./255)
val_gen = ImageDataGenerator( rescale = 1./255)
test_gen = ImageDataGenerator( rescale = 1./255)


# In[5]:


# flow_from_directory: 폴더에서 이미지 가져오기
# 폴더명, 이미지 크기, 한번에 변환 할 이미지 수, 라벨링 모드
# 이진분류 = binary, 다중 분류 = categorical 
# 라벨 번호는 0부터 시작(cat은 0, dog는 1)
train_generator = train_gen.flow_from_directory(train_dir,
                              target_size =(64,64),
                              batch_size=32,
                              class_mode = 'binary')
val_generator = val_gen.flow_from_directory( validation_dir,
                            target_size = (64,64),
                            batch_size=32,
                            class_mode =  'binary')
test_generator = test_gen.flow_from_directory( test_dir,
                            target_size = (64,64),
                            batch_size=32,
                            class_mode =  'binary')


# In[ ]:


c_model = Sequential()

# 입력층(CNN)
# 특징을 도드라지게 해준다
c_model.add(Conv2D(filters = 32, # 사진에서 찾을 특성 개수
                   kernel_size = (3,3), # 한번에 확인할 픽셀의 수
                   input_shape = (64,64,3), # 입력 데이터의 크기
                   padding = 'same', # 가장 자리를 0으로 채움 
                  # same : 입력데이터의 크기와 동일하게 맞춰준다
                   activation = 'relu'))
#불필요한 부분 삭제
c_model.add(MaxPooling2D(pool_size = (2,2),strides=2))

c_model.add(Conv2D(filters = 32, # 사진에서 찾을 특성 개수
                   kernel_size = (3,3), # 한번에 확인할 픽셀의 수
                   input_shape = (64,64,3), # 입력 데이터의 크기
                   padding = 'same', # 가장 자리를 0으로 채움(입력이미지크기=출력이미지크기)
                  # same : 입력데이터의 크기와 동일하게 맞춰준다
                   activation = 'relu'))

c_model.add(MaxPooling2D(pool_size = (2,2),strides=2))

# 1차원으로 데이터 축소
c_model.add(Flatten())

#은닉층
c_model.add(Dense(units=128, activation = 'relu'))

#출력층
c_model.add(Dense(units=1, activation= 'sigmoid'))

c_model.summary()


# In[8]:


adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
c_model.compile(loss='binary_crossentropy',
                optimizer = adam,
                metrics=['accuracy'])
history = c_model.fit_generator(generator=train_generator,
            steps_per_epoch=200,
            epochs=30,
            validation_data = val_generator,
            validation_steps = 2)


# In[9]:



plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# In[ ]:




