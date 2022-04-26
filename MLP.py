import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def run():
    start = time.time();
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

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), max_iter=700)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    
    end = time.time()
    print("Time elapsed:")
    print(end - start)

    plot_confusion_matrix(classifier, X_test, y_test)
    plt.show()
    
run()
