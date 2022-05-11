import pandas as pd

fish = pd.read_csv('http://bit.ly/fish_csv_data')
print(pd.unique(fish['Species']))

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.classes_)

print(kn.predict(test_scaled[:5]))

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

import numpy as np
import matplotlib.pyplot as plt
z = np.arrange(-5,5,0.1)
phi = 1/(1+np.exp(-z))
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

char_arr = np.array(['A','B','C','D','E'])
print(char_arr[[True, False, True, False, False]])