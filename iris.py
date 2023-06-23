import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from collections import Counter


df = pd.read_csv("C:\CIDSA\Iris.csv")

X = df.drop('Species', axis=1)
Y = df['Species']
print(X)
print(Y)

'''
irs = load_iris()
print(irs)
print(irs.keys())
print(irs.data)
print(irs.target)
print(irs.feature_names)
print(irs.target_names)
print(irs.DESCR)

df = pd.read_csv("C:\CIDSA\Iris.csv")
print(df)
print(df.head(10))
print(df.tail)
print(df.columns.values)
print(df.describe())



#feature
bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featuresScore = pd.concat([dfcolumns, dfscores], axis=1)
featuresScore.columns=['Specs','Score']

print(featuresScore)

model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)

feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.nlargest(4).plot(kind='pie')
plt.show()



#counter
X = df.drop('species', axis=1)
Y = df['species']
print(Y)

print(df.isnull().sum())
print(df.notnull().sum())
print(Counter(Y))
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))




#log reg
logr = LogisticRegression()
pca = PCA(n_components=2)
X = df.drop('species', axis=1)
Y= df['species']
pca.fit(X)
X = pca.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=2, test_size=0.4)
logr.fit(X_train, Y_train)
Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred))




#boxplot
sns.boxplot(df['SepalLengthCm'])
plt.show()
print(df['SepalLengthCm'])
Q1 = df['SepalLengthCm'].quantile(0.25)
Q3 = df['SepalLengthCm'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
upper = Q3 + 1.5*IQR
lower = Q1 + 1.5*IQR
print(upper)
print(lower)
out1 = df['SepalLengthCm'] < lower.values
out1 = df['SepalLengthCm'] < upper.values
df['SepalLengthCm'].replace(out1, lower, inplace = True)
df['SepalLengthCm'].replace(out2, upper, inplace = True)
print(df['SepalLengthCm'])



#randomforest
rf = RandomForestClassifier()
df['SepalLengthCm']=pd.cut(df['SepalLengthCm'],3,labels=['0','1','2'])
df['SepalWidthCm']=pd.cut(df['SepalWidthCm'],3,labels=['0','1','2'])
df['PetalLengthCm']=pd.cut(df['PetalLengthCm'],3,labels=['0','1','2'])
df['PetalWidthCm']=pd.cut(df['PetalWidthCm'],3,labels=['0','1','2'])

print(df)



#logistic regression
logr = LogisticRegression()
pca = PCA(n_components=2)
X = df.drop('Species', axis=1)
Y= df['Species']
pca.fit(X)
X = pca.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=2, test_size=0.4)
logr.fit(X_train, Y_train)
Y_pred = logr.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
'''
