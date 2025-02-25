TRAIN_PATH = './A1_DATASET/train.txt'
VAL_PATH = './A1_DATASET/val.txt'
class Data():
    def get_train(self):
        with open(TRAIN_PATH) as file:
            data = file.read()
        print(data)
        return data

    def get_val(self):
        with open(VAL_PATH) as file:
            data = file.read()
        return data