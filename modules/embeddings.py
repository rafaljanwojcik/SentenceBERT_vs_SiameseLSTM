import numpy as np
import wget
import os 
import zipfile
import torch
from logging import Logger

class EmbeddedVocab():
  def __init__(self, embeddings_path: str, embedding_size: int, download: bool, logdir: str, logger: Logger):
    self.embedding_size = embedding_size
    self.embeddings_path = embeddings_path
    self.logger = logger
    
    if download:
      path = logdir/'embeddings/'
      if not path.exists(): 
        os.mkdir(path)

      self.logger.info("Downloading pretrained Glove embeddings from: http://nlp.stanford.edu/data/glove.6B.zip to: {}".format(path))
      url = 'http://nlp.stanford.edu/data/glove.6B.zip'
      wget.download(url, str(path))

      self.logger.info('Extracting embeddings...')
      with zipfile.ZipFile(path/'glove.6B.zip', 'r') as zip_ref:
        zip_ref.extractall(path)
        self.logger.info('Extracted.')
      self.embeddings_path = path/'glove.6B.100d.txt'
      
    
    
    self.embeddings, self.vocabulary, self.reverse_vocab = self.initiate_vocab_and_embeddings()
    
        
  def initiate_vocab_and_embeddings(self):
    self.logger.info('Initializing embeddings vocab...')
    with open(self.embeddings_path, encoding="utf8") as f:
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
    self.logger.info('Embeddings initialized with {} words, with {} embedding dimensions'.format(len(embeddings)-1, self.embedding_size))
    embeddings = torch.from_numpy(np.array(list(embeddings.values())))
    return(embeddings, vocab, reverse_vocab)
  
  def convert_embeddings(self):
    return torch.from_numpy(np.array(list(self.embeddings.values()), dtype='float64'))