from pytorch_fast_elmo import FastElmoWordEmbedding, load_and_build_vocab2id, batch_to_word_ids
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from torch import masked_select

vocab_file = '../vocabulary/tokens.txt'
num_tokens = 793471
embedding_dim = 256
num_samples = 20
options = {
            'options_file': None, 
            'weight_file': None, 
            'word_embedding_cnt': num_tokens,
            'exec_managed_lstm_bos_eos': False,  
            'word_embedding_requires_grad': True, 
            'forward_lstm_requires_grad': True, 
            'backward_lstm_requires_grad': True,
            'word_embedding_dim': embedding_dim,
            'lstm_input_size': embedding_dim,
            'lstm_hidden_size': embedding_dim,
            'lstm_cell_size': 4 * embedding_dim
}

### build model
vocab2id = load_and_build_vocab2id(vocab_file)
elmo = FastElmoWordEmbedding(**options)
classifier = SampledSoftmaxLoss(num_tokens, 2 * embedding_dim, num_samples)

### embed demo sentences
sentences = [['First', 'sentence', '.'], ['Another', '.'], ['I', 'will', 'give', 'power', 'to', 'them', 'so', 'they', 'can', 'continue', 'their', 'mission', '.']]
word_ids = batch_to_word_ids(sentences, vocab2id)
embeddings = elmo(word_ids)

mask = embeddings["mask"].bool()
mask[:,0] = False
targets = masked_select(word_ids, mask)
context = embeddings["elmo_representations"][0][mask.roll(-1,1)]

loss = classifier(context, targets)
print(loss)
loss.backward()
