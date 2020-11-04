from glob import glob
from torch.utils.data import Dataset, IterableDataset, ConcatDataset, ChainDataset

class WordDataset(Dataset):
    def __init__(self, filename):
        with open(filename) as file:
            self.sentences = file.readlines()
        
    def __getitem__(self, index):
        return self.sentences[index].strip("\n").split()
        
    def __len__(self):
        return len(self.sentences)


class WordIterableDataset(IterableDataset):
    def __init__(self, filename):
        self.filename = filename
        
    def __iter__(self):
        with open(self.filename) as file:
            for line in file:
                yield line.strip("\n").split()
        
    def __len__(self):
        return len(self.sentences)


class OneBillionWordDataset(ConcatDataset):
    def __init__(self, pathname):
        shards = glob(pathname)   
        super().__init__([WordDataset(shard) for shard in shards])


class OneBillionWordIterableDataset(ChainDataset):
    def __init__(self, pathname):
        shards = glob(pathname)   
        super().__init__([WordIterableDataset(shard) for shard in shards])
