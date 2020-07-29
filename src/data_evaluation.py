import numpy as np
from src.preprocess import process, load


def search_near_stat(_stat, _idx, _tol):
    result = []
    compare = _stat[_idx]
    for i, s in enumerate(_stat):
        if i == _idx:
            continue
        v = np.sum(np.square(s - compare))

        if v < _tol:
            result.append(i)

    return result


if __name__ == '__main__':
    header = ['season', 'name', 'age']
    pos = ['SP', 'RP', '1B', '2B', '3B', 'SS', 'RF', 'CF', 'LF', 'C', 'DH', 'OF', 'P']
    data_train, data_cv, data_test, data_raw_test = load('./data/fa_data_2012_2019_[5.0, 3.0, 2.0]_mod.npy')
    (X_train, y_train), (X_cv, y_cv), (X_test, y_test), scaler = process(data_train, data_cv, data_test)

    target = list(data_raw_test[:, 1]).index('Anthony Rendon')
    stat = X_test[:, len(header) + len(pos) - 1:]

    nearby = search_near_stat(stat, target, 2)
    print(data_raw_test[nearby, 1])
