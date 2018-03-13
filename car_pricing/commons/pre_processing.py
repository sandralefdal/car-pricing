from sklearn.preprocessing import MinMaxScaler


def scale(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)