import config
import random
import torch
from torch.cuda import is_available
device = torch.device("cuda" if is_available() else "cpu")


class Batch:
    def __init__(self):
        self.nl = None
        self.ast = None
        self.code = None
        self.nl_ex = None
        self.code_ex = None
        self.code_oovs = None


class BatchQueue:
    def __init__(self, batch_sz, key="train"):
        self.key = key
        self.batch_sz = batch_sz
        self.__ast_vocab = self.__read_vocab(config.AST_VOCAB)
        self.__code_vocab = self.__read_vocab(config.CODE_VOCAB)
        self.__nl_vocab, self.nl_list = self.__read_vocab(config.NL_VOCAB, return_data=True)

        self.__nl_data = self.__read_data(config.paths[self.key]["NL_INPUT"])
        self.__ast_data = self.__read_data(config.paths[self.key]["AST_INPUT"])
        self.__code_data = self.__read_data(config.paths[self.key]["CODE_INPUT"])
        
        self.__fill_UNK = lambda lst, vocab: [vocab["<UNK>"] if w >= len(vocab) else w for w in lst]
        self.post_pad = lambda seqs, maxl: [(x + [0]*(maxl-len(x))) for x in seqs]

    def __read_vocab(self, input_path, return_data=False):
        with open(input_path, 'r') as content_file:
            data = content_file.read().split('\n')
        vocab_dict = {k: v for (v, k) in enumerate(data, start=1)}
        if return_data:
            return vocab_dict, data
        return vocab_dict

    def __read_data(self, input_path):
        with open(input_path, 'r') as input_file:
            data = input_file.readlines()
        return data

    def __helper(self, i, j):
        code_oovs = []
        nl_batch = []
        ast_batch = []
        code_batch = []
        ex_code_batch = []
        ex_nl_batch = []
        l1, l2, l3 = 0, 0, 0
        
        for line in self.__code_data[i:j]:
            idxs_ex = []
            line = f"<S> {line} </S>"
            for word in line.split():
                try:
                    idxs_ex.append(self.__code_vocab[word])
                except KeyError:
                    if word not in code_oovs:
                        code_oovs.append(word)
                    idxs_ex.append(len(self.__code_vocab) + code_oovs.index(word))
            l1 = max(l1, len(idxs_ex))
            ex_code_batch.append(idxs_ex)
            code_batch.append(self.__fill_UNK(idxs_ex, self.__code_vocab))

        for line in self.__ast_data[i:j]:
            line = f"<S> {line} </S>"
            idxs = [self.__ast_vocab[w] for w in line.split()]
            l2 = max(l2, len(idxs))
            ast_batch.append(idxs)

        for line in self.__nl_data[i:j]:
            idxs_ex = []
            line = f"<S> {line} </S>"
            for word in line.split():
                try:
                    idxs_ex.append(self.__nl_vocab[word])
                except KeyError:
                    try:
                        idxs_ex.append(len(self.__nl_vocab) + code_oovs.index(word)) 
                    except ValueError:
                        idxs_ex.append(self.__nl_vocab["<UNK>"])
            l3 = max(l3, len(idxs_ex))
            ex_nl_batch.append(idxs_ex)
            nl_batch.append(self.__fill_UNK(idxs_ex, self.__nl_vocab))
        
        l = max(l1, l2)
        nl_batch = self.post_pad(nl_batch, l3)
        ex_nl_batch = self.post_pad(ex_nl_batch, l3)
        ast_batch = self.post_pad(ast_batch, l)
        code_batch = self.post_pad(code_batch, l)
        ex_code_batch = self.post_pad(ex_code_batch, l)
        
        batch = Batch()
        batch.code_oovs = code_oovs
        batch.nl = torch.Tensor(nl_batch).long().to(device)
        batch.ast = torch.Tensor(ast_batch).long().to(device)
        batch.code = torch.Tensor(code_batch).long().to(device)
        batch.nl_ex = torch.Tensor(ex_nl_batch).long().to(device)
        batch.code_ex = torch.Tensor(ex_code_batch).long().to(device)
        return batch

    def batcher(self, shuffle=True):
        if shuffle:
            combined = list(zip(self.__code_data, self.__ast_data, self.__nl_data))
            random.shuffle(combined)
            self.__code_data, self.__ast_data, self.__nl_data = zip(*combined)

        i, l = 0, len(self.__code_data)
        while i + self.batch_sz < l:
            j = i + self.batch_sz
            yield self.__helper(i, j)
            i = j