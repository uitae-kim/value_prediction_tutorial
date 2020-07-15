import sys
import warnings
from src.preprocess import *
from src.model_HW import *


# suppress warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# target
if len(sys.argv) == 1:
    path = './data/fa_data_2012_2019_[5.0, 3.0, 2.0].npy'
else:
    if 'p' in sys.argv[1].lower():
        path = './data/fa_data_2012_2019_[5.0, 3.0, 2.0]_pit.npy'
    else:
        path = './data/fa_data_2012_2019_[5.0, 3.0, 2.0]_bat.npy'

# data
train, cv, test, raw = load(path)
(X_train, y_train), (X_cv, y_cv), (X_test, y_test), scaler = process(train, cv, test)

# model
network, error, output = model([1024, 512, 256, 128], X_train.shape[1])
pred = run(network, error, output, X_train, y_train, X_test, y_test)
pred = scaler.inverse_transform(np.concatenate((X_test, pred), axis=1))[:, -1]
