import torch
import torch.nn as nn
from embeddings import EmbeddedVocab

class MaLSTM(nn.Module):
  '''
  architechture follows example proposed by javiersuweijie from fast.ai forum blogpost:
  https://forums.fast.ai/t/siamese-network-architecture-using-fast-ai-library/15114/3
  '''
  def __init__(self, hidden_size: int, pretrained_embeddings: EmbeddedVocab, embedding_dim: int, num_layers: int, n_token: int, train_embeddings: bool = True, use_pretrained:bool = False,  dropouth: float=0.3):
      super(MaLSTM, self).__init__()
      self.init_range=0.1
      if use_pretrained:
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=not train_embeddings)
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.weight.requires_grad = train_embeddings
      else:
        self.embedding = nn.Embedding(n_token, embedding_dim, padding_idx=0)
        self.embedding.weight.data.uniform_(-self.init_range, self.init_range)
      
      self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropouth)
      self.lstm = self.lstm.float()
      
  def forward(self, inputs):
    
    embedded1 = self.embedding(inputs[0])
    embedded2 = self.embedding(inputs[1])
    
    outputs1, hiddens1 = self.lstm(embedded1)
    outputs2, hiddens2 = self.lstm(embedded2)

    return self.similarity(outputs1, outputs2)

  def similarity(self, h1, h2):
    return torch.exp(-torch.norm(h1-h2, dim=(2,1)))