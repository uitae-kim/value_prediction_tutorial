import tensorflow as tf
import numpy as np

_scaler = None
_X_test = None
_y_test = None
_X_raw = None


def _set_scaler():
    global _scaler, _X_test, _X_raw, _y_test
    from src.preprocess import process, load

    data_train, data_cv, data_test, data_raw_test = load('./data/fa_data_2012_2019_[5.0, 3.0, 2.0]_mod.npy')
    (X_train, y_train), (X_cv, y_cv), (X_test, y_test), scaler = process(data_train, data_cv, data_test)

    _scaler = scaler
    _X_test = X_test
    _y_test = data_test[:, -1]
    _X_raw = data_raw_test


def predict(data, path):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(path)
        saver.restore(sess, path[:-5])

        graph = sess.graph
        X = graph.get_tensor_by_name('Placeholder:0')
        out = graph.get_tensor_by_name('transpose:0')

        result = sess.run(out, feed_dict={X: data})

        return _scaler.inverse_transform(np.concatenate((data, np.transpose(result)), axis=1))[:, -1]


if __name__ == '__main__':
    _set_scaler()

    data = np.load('./data/fa_data_2012_2019_[5.0, 3.0, 2.0]_mod.npy', allow_pickle=True)
    _X_raw = data
    _y_test = data[:, -1]
    idx = [x for x in range(data.shape[1])]
    idx.remove(1)
    data = _scaler.transform(data[:, idx])
    _X_test = data[:, :-1]

    path = './checkpoints/4096/4096-99.meta'
    names = _X_raw[:, 1]
    price = predict(_X_test, path)
    # price = price / 1000000

    header = ['season', 'name', 'age']
    pos = ['SP', 'RP', '1B', '2B', '3B', 'SS', 'RF', 'CF', 'LF', 'C', 'DH', 'OF', 'P']
    position = []

    for raw in _X_raw:
        for i, p in enumerate(raw[len(header):len(header) + len(pos)]):
            if p == 1:
                position.append(pos[i])
                break
    position = np.reshape(np.asarray(position), (len(position), 1))

    result = np.concatenate(
        (np.reshape(names, (len(names), 1)), np.reshape(price, (len(price), 1)), position,
         np.reshape(_y_test, (len(_y_test), 1))),
        axis=1)

    print(result)

    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(_y_test, 'r')
    ax1.plot(result[:, 1], 'g')
    plt.waitforbuttonpress()

    with open('./data/result_all.csv', 'w') as f:
        for r in result:
            # name / predicted / contract / position
            f.write(f'{r[0]},{r[1]/1000000},{r[3]/1000000},{r[2]}\n')
