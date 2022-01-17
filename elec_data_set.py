from pandas import read_csv
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_data():
    dataset = read_csv(r'train.txt', delimiter=';', decimal=',', header=0)
    # summarize
    dataset = dataset.drop('Unnamed: 0', axis=1)

    arr = np.array(dataset)
    rows = arr.shape[0]

    del dataset

    y_data = []
    for i in range(int(rows/4)):
        y_data.append(np.mean(arr[i*4:(i+1)*4, :], axis=0))

    y_data = np.array(y_data)
    #return np.diff(y_data, axis=0)
    scaler = MinMaxScaler()
    scaler = scaler.fit(y_data)

    train_n = scaler.transform(y_data)
	
    return train_n, scaler
