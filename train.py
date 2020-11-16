from data import OneBillionWordIterableDataset
from torch.utils.data import DataLoader
from pytorch_fast_elmo import FastElmoWordEmbedding, load_and_build_vocab2id, batch_to_word_ids
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from torch import device
from torch.optim import Adagrad


#options
dataset_path = '../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*'
vocab_file = '../vocabulary/tokens.txt'
num_tokens = 793471
embedding_dim = 64
num_samples = 10
batch_size = 128
device = device('cuda')
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
            'lstm_cell_size': 8 * embedding_dim,
            'output_representation_dropout': 0.1
}

### build model
vocab2id = load_and_build_vocab2id(vocab_file)
elmo = FastElmoWordEmbedding(**options)
classifier = SampledSoftmaxLoss(num_tokens, embedding_dim, num_samples)

### move model to device
elmo.to(device)
classifier.to(device)

### set up training
dataset = OneBillionWordIterableDataset(dataset_path)
optimizer = Adagrad(list(elmo.parameters()) + list(classifier.parameters()), lr=0.2, initial_accumulator_value=1.0)

for i, batch in enumerate(DataLoader(dataset, batch_size=batch_size, collate_fn=list), 0):
    optimizer.zero_grad()
    word_ids = batch_to_word_ids(batch, vocab2id).to(device) # TODO: create tensor directly on device
    embeddings = elmo(word_ids)
    
    mask = embeddings["mask"].bool()
    mask[:,0] = False
    mask_rolled = mask.roll(-1,1)

    targets_forward = word_ids[mask]
    targets_backward = word_ids[mask_rolled]
    context_forward = embeddings["elmo_representations"][0][:,:,:embedding_dim][mask_rolled]
    context_backward = embeddings["elmo_representations"][0][:,:,embedding_dim:][mask]

    loss_forward = classifier(context_forward, targets_forward) / targets_forward.size(0)
    loss_backward = classifier(context_backward, targets_backward) / targets_backward.size(0)
    loss = 0.5 * loss_forward + 0.5 * loss_backward

    loss.backward()
    optimizer.step()
    print('[%6d] loss: %.3f' % (i + 1, loss.item()))
