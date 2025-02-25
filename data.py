class Data:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_contents(self) -> str:
        with open(self.file_path, 'r') as file:
            return file.read()