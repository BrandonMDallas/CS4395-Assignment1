def get_train():
    train_path = './A1_DATASET/train.txt'
    with open(train_path) as file:
        data = file.read()
    return data

def get_val():
    val_path = './A1_DATASET/val.txt'
    with open(val_path) as file:
        data = file.read()
    return data