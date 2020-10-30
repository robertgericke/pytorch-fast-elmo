import glob
from torch.utils.data import Dataset, DataLoader
from pytorch_fast_elmo import FastElmoWordEmbedding, load_and_build_vocab2id, batch_to_word_ids
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from torch import masked_select
from torch.optim import SGD

class OneBillionWordDataset(Dataset):
    def _load_shard(self, shard_name):
        with open(shard_name) as file:
            self.sentences = file.readlines()  
    def __init__(self):
        shards = glob.glob("../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*")
        self._load_shard(shards[0])
    def __getitem__(self, index):
        return self.sentences[index].strip("\n").split()
    def __len__(self):
        return len(self.sentences)
 

#options
vocab_file = '../vocabulary/tokens.txt'
num_tokens = 793471
embedding_dim = 256
num_samples = 10
batch_size = 10
options = {
            'options_file': None, 
            'weight_file': None, 
            'exec_managed_lstm_bos_eos': False,  
            'word_embedding_requires_grad': True, 
            'forward_lstm_requires_grad': True, 
            'backward_lstm_requires_grad': True,
            'word_embedding_cnt': num_tokens,
            'word_embedding_dim': embedding_dim,
            'lstm_input_size': embedding_dim,
            'lstm_hidden_size': embedding_dim,
            'lstm_cell_size': 4 * embedding_dim,
            'output_representation_dropout': 0.1
}

### build model
vocab2id = load_and_build_vocab2id(vocab_file)
elmo = FastElmoWordEmbedding(**options)
classifier = SampledSoftmaxLoss(num_tokens, 2 * embedding_dim, num_samples)

### set up training
dataset = OneBillionWordDataset()
optimizer = SGD(list(elmo.parameters()) + list(classifier.parameters()), lr=0.001, momentum=0.9)

for i, batch in enumerate(DataLoader(dataset, batch_size=batch_size, collate_fn=list), 0):
    optimizer.zero_grad()
    word_ids = batch_to_word_ids(batch, vocab2id)
    embeddings = elmo(word_ids)
    mask = embeddings["mask"].bool()
    mask[:,0] = False
    targets = masked_select(word_ids, mask)
    context = embeddings["elmo_representations"][0][mask.roll(-1,1)]
    loss = classifier(context, targets)
    loss.backward()
    optimizer.step()
    print('[%6d] loss: %.3f' % (i + 1, loss.item()))
