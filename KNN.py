import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

start = time.time();

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

filename = "TRAININGDATAEDITED.csv"
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
# Assign column names to the dataset
names = ['actor_account','login_count','logout_count','login_day_count','play_time','avg_money','ip_count','max_level','login_total_day_x','playtime_per_day','sit_count','exp_get_amout','item_get_count','money_get_count','killed_by_pc','killed_by_npc','teleport_count','reborn_count','question_count','login_total_day_y','sit_count_perday','item_get_count_perday','money_get_count_perday','teleport_count_perday','total_party_time','average_party_time','class']

# Read dataset to pandas dataframe
dataset = pd.read_csv(filename, names=names)
dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset = clean_dataset(dataset)

print(dataset.head())

X = dataset[['login_count','logout_count','login_day_count','play_time','avg_money','ip_count','max_level','login_total_day_x','playtime_per_day','sit_count','exp_get_amout','item_get_count','money_get_count','killed_by_pc','killed_by_npc','teleport_count','reborn_count','question_count','login_total_day_y','sit_count_perday','item_get_count_perday','money_get_count_perday','teleport_count_perday','total_party_time','average_party_time',]] 
y = dataset['class'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
#this part works
print(X_train)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

end = time.time()
print("Time elapsed:")
print(end - start)
