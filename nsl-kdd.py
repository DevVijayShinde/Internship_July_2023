import pandas as pd
df = pd.read_arff("C:\CIDSA\KDDTrain+_20Percent.arff")

X = df.drop('duration', axis=1)
Y = df['duration']
print(X)
print(Y)