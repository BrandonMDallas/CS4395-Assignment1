class Data:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_contents(self) -> str:
        with open(self.file_path, 'r') as file:
            return file.read()
'''
TRAIN_PATH = './A1_DATASET/train.txt'
VAL_PATH = './A1_DATASET/val.txt'
class Data():
    def get_train(self):
        with open(TRAIN_PATH) as file:
            data = file.read()
        return data

    def get_val(self):
        with open(VAL_PATH) as file:
            data = file.read()
        return data
'''