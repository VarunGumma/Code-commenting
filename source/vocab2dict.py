from config import *


class VocabData:
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as content_file:
            data = content_file.read().split('\n')
            content_file.close()
        self.vocab_dict = {k: v for (v, k) in enumerate(data)}
