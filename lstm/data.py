import pandas as pd
import numpy as np
from re import sub
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class ImportData:
  def __init__(self, datasetpath: str):
    self.data = pd.read_csv(datasetpath).dropna()[['question1', 'question2', 'is_duplicate']]
    
  def train_test_split(self, seed: int=44, test_size: int=40000):
    self.train, self.test = train_test_split(self.data, test_size=test_size, random_state=seed)
    
  def __getitem__(self, idx: int):
    ex = self.data.loc[idx]
    return ex.question1, ex.question2, ex.is_duplicate
  
  def __len__(self):
    return self.data.shape[0]


class QuoraQuestionDataset(Dataset):
  def __init__(self, datasetvar: ImportData, use_pretrained_emb: bool=False, reverse_vocab: dict = None, preprocess: bool = True, train: bool = True):
    self.use_cuda = torch.cuda.is_available()
    self.data = datasetvar.copy()

    if preprocess == True:
      self.preprocessing()  

    if not use_pretrained_emb and train:
      unique_words = self.data.question1.str.split(' ').append(self.data.question2.str.split(' '))
      unique_words = pd.Series([i for j in unique_words.values for i in j]).unique().tolist()
      unique_words.insert(0, 'pad')
      self.unique_words = len(unique_words)
      self.reverse_vocab = dict(zip(unique_words, range(0,219673)))

    elif type(reverse_vocab) == dict:
      self.reverse_vocab = reverse_vocab
      self.unique_words = len(reverse_vocab.keys())


    else:
      raise Exception("Invalid reverse_vocab arg (cannot create dictionary with mapping of words to their indices).")
  
  def preprocessing(self, reverse_vocab):
    self.data.question1 = self.data.question1.apply(lambda x: self.text_to_word_list(x))
    self.data.question2 = self.data.question2.apply(lambda x: self.text_to_word_list(x))
    
  def words_to_ids(self):
    self.data.question1 = self.data.question1.apply(lambda x: list(map(lambda y: self.replace_words(y, self.reverse_vocab), x.split())))
    self.data.question2 = self.data.question2.apply(lambda x: list(map(lambda y: self.replace_words(y, self.reverse_vocab), x.split())))
      
    #self.data.is_duplicate = self.data.is_duplicate.apply(lambda x: np.array([x], dtype='int8'))
    
    
  def text_to_word_list(self, text: str):
    ''' Pre process 
    method from: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb'''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = sub(r"[^A-Za-z0-9^,!.\/'+-=)(]", " ", text)
    text = sub(r"what's", "what is ", text)
    text = sub(r"\'s", " ", text)
    text = sub(r"\'ve", " have ", text)
    text = sub(r"can't", "cannot ", text)
    text = sub(r"n't", " not ", text)
    text = sub(r"i'm", "i am ", text)
    text = sub(r"\'re", " are ", text)
    text = sub(r"\'d", " would ", text)
    text = sub(r"\'ll", " will ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\/", " ", text)
    text = sub(r"\^", " ^ ", text)
    text = sub(r"\+", " + ", text)
    text = sub(r"\-", " - ", text)
    text = sub(r"\=", " = ", text)
    text = sub(r"'", " ", text)
    text = sub(r"(\d+)(k)", r"\g<1>000", text)
    text = sub(r":", " : ", text)
    text = sub(r" e g ", " eg ", text)
    text = sub(r" b g ", " bg ", text)
    text = sub(r" u s ", " american ", text)
    text = sub(r"\0s", "0", text)
    text = sub(r" 9 11 ", "911", text)
    text = sub(r"e - mail", "email", text)
    text = sub(r"j k", "jk", text)
    text = sub(r"\s{2,}", " ", text)

    return text  
   
  #def spell_checker(self, word):
  
  def replace_words(self, word: str, reverse_vocab: dict):
    if word in reverse_vocab.keys():
      return reverse_vocab[f'{word}']
    else:
      return 0
    
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
      
    ex = self.data.iloc[idx] 

    if type(idx)==list:
        return(ex.question1.values, ex.question2.values, ex.is_duplicate.values)
    else:
        return (ex.question1, ex.question2, ex.is_duplicate)
    
  def __len__(self):
    return self.data.shape[0]