import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import *

matplotlib.rcParams['axes.unicode_minus'] = False
sns.set(font="Times New Roman", style="ticks", font_scale=1.1)
pd.set_option("max_colwidth", 200)

index_random_x = np.arange(1, 122)
for i in range(500):
    np.random.shuffle(index_random_x)
c1 = index_random_x[0:99]
c2 = index_random_x[99:122]
with open(r"D:\Code\am_gdm\特征重要性分析\data_knn2.csv", 'r') as df1:
    reader = csv.reader(df1)
    rows = list(reader)
    df2 = []
    df3 = []
    for index, info in enumerate(rows):
        df2.append(info[:1])
        df3.append(info[1:])
y_train = []
y_val = []
for i in c1:
    y_train.append(df2[i])
for j in c2:
    y_val.append(df2[j])
X_train = []
X_val = []
for i in c1:
    X_train.append(df3[i])
for j in c2:
    X_val.append(df3[j])
train_x = rows[0][1:]
rfc1 = RandomForestClassifier(n_estimators=100,
                              max_depth=5,
                              oob_score=True,
                              class_weight="balanced",
                              random_state=1)
rfc1.fit(X_train, y_train)
rfc1_lab = rfc1.predict(X_train)
rfc1_pre = rfc1.predict(X_val)
importances = pd.DataFrame({"feature": train_x,
                            "importance": rfc1.feature_importances_})
importances = importances.sort_values("importance", ascending=True)
rowsfe = importances.loc[:, 'feature']
rowsim = importances.loc[:, 'importance']
index_in = []
value_fe = []
value_im = []
for value in rowsfe.keys():
    index_in.append(value)
for value in rowsfe:
    value_fe.append(value)
for value in rowsim:
    value_im.append(value)
pd.DataFrame(index_in).to_csv(r"D:\Code\am_gdm\特征重要性分析\1_mean.xlsx", index=False)
pd.DataFrame(value_fe).to_csv(r"D:\Code\am_gdm\特征重要性分析\2_mean.xlsx", index=False)
pd.DataFrame(value_im).to_csv(r"D:\Code\am_gdm\特征重要性分析\3_mean.xlsx", index=False)

length = len(importances)
importances.iloc[length-30:length].plot(kind="barh", figsize=(10,6), x="feature", y="importance", legend=False)
plt.xlabel("Importance Score")
plt.ylabel("")
plt.grid()
plt.show()


