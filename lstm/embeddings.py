import numpy as np
import wget
import os 
import zipfile
import torch

class EmbeddedVocab():
  def __init__(self, embeddings_path, embedding_size, download=False, logdir='/logs'):
    self.embedding_size = embedding_size
    self.embeddings_path = embeddings_path
    
    if download:
      print("Downloading pretrained Glove embeddings from: http://nlp.stanford.edu/data/glove.6B.zip")
      url = 'http://nlp.stanford.edu/data/glove.6B.zip'
      wget.download(url, os.getcwd()+logdir)
      with zipfile.ZipFile(os.getcwd()+logdir, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd()+logdir)
      self.embeddings_path = os.getcwd()+logdir+filename
      
    
    
    self.embeddings, self.vocabulary, self.reverse_vocab = self.initiate_vocab_and_embeddings()
    
        
  def initiate_vocab_and_embeddings(self):
    f = open(self.embeddings_path)
    #embeddings = np.zeros((len(f),self.embedding_size))
    embeddings = {0:np.array([0]*self.embedding_size)}
    vocab = {'PAD':0}
    reverse_vocab = {0:'PAD'}
    i=1
    for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings[i] = coefs
      vocab[i] = word
      reverse_vocab[word] = i
      i+=1
    f.close()
    print(f'Vocab initialized with {len(embeddings)-1} words, with {self.embedding_size} embedding dimensions')
    embeddings = torch.from_numpy(np.array(list(embeddings.values())))
    return(embeddings, vocab, reverse_vocab)
  
  def convert_embeddings(self):
    return torch.from_numpy(np.array(list(self.embeddings.values()), dtype='float64'))