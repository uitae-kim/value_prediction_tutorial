def parse(path):
    train = []
    eval = []
    test = []
    with open(path) as f:
        while True:
            line = f.readline()
            if line == '':
                break

            if 'Epoch' not in line:
                continue

            train.append(line.split('Train Error:')[1].split()[0])
            eval.append(line.split('Cross Validation Error:')[1].split()[0])
            test.append(line.split('Test Error:')[1].split()[0])

    save(train, eval, test, path + '.csv')

def save(train, eval, test, path):
    with open(path, 'w') as f:
        for i in range(len(train)):
            f.write(f'{train[i]},{eval[i]},{test[i]}\n')


if __name__ == '__main__':
    parse('./data/colab_result_all_512_noreg')
    parse('./data/colab_result_all_512_reg')
