import pandas as pd

data = pd.Series([1, 2, 3, 4, 5, "A", "B", '가'],
                 ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
print(data)
print(type(data))

data2 = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})

print(data2)
print(type(data2))

