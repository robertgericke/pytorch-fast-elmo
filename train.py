from data import OneBillionWordIterableDataset
from torch.utils.data import DataLoader
from pytorch_fast_elmo import FastElmoWordEmbedding, load_and_build_vocab2id, batch_to_word_ids
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from torch import device
from torch.optim import Adagrad
from torch.cuda import device_count
import torch.multiprocessing as mp
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import torch

#options
dataset_path = '../1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*'
vocab_file = '../vocabulary/tokens.txt'
num_tokens = 793471
embedding_dim = 16
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

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

def train(rank, world_size):
    ### build model
    vocab2id = load_and_build_vocab2id(vocab_file)
    elmo = FastElmoWordEmbedding(**options)
    classifier = SampledSoftmaxLoss(num_tokens+1, embedding_dim, num_samples) # +1 for padding token id 0

    ### move model to device
    elmo.to(device)
    classifier.to(device)
    
    elmo = DDP(elmo, device_ids=[rank])
    classifier = DDP(classifier) # , device_ids=[rank]

    ### set up training
    dataset = OneBillionWordIterableDataset(dataset_path)
    optimizer = Adagrad(list(elmo.parameters()) + list(classifier.parameters()), lr=0.2, initial_accumulator_value=1.0)
    
    for i, batch in enumerate(DataLoader(dataset, batch_size=batch_size, collate_fn=list), 1):
        optimizer.zero_grad()
        word_ids = batch_to_word_ids(batch, vocab2id, device=device)
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
        print('[%6d] loss: %.3f' % (i, loss.item()))


def init_process(rank, world_size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    fn(rank, world_size)


if __name__ == "__main__":
    world_size = 2
    mp.spawn(init_process, args=(world_size, train), nprocs=world_size, join=True)