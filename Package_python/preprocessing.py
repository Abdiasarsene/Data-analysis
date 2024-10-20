# preprocessing.py
from sklearn.preprocessing import StandardScaler

def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
