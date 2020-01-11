import argparse
import torch
import os
from time import time

from data import ImportData, QuoraQuestionDataset
from embeddings import EmbeddedVocab
from model import MaLSTM
from utils import generate_batch, train, eval

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--logdir", type=str, help="Directory to save all downloaded files, and model checkpoints.", default=os.getcwd()+'/logs/')  
    parser.add_argument("-df", "--data_file", type=str, help="Path to dataset.", default="../logs/questions.csv")
    parser.add_argument("-e", "--use_pretrained", type=bool, help="Boolean, whether use pretrained embeddings.", default=False)
    parser.add_argument("-e", "--download_emb", type=bool, help="Bool, whether to download embeddings or not (default is to download 100dimensional Glove embeddings)", default=True)
    parser.add_argument("-e", "--emb_dim", type=int, help="Dimensions of pretrained embeddings", default=100)
    parser.add_argument("-e", "--emb_path", type=str, help="path to file with pretrained embeddings", default=os.getcwd()+'/logs/'+'glove.6B.100d.txt')
    parser.add_argument("-z", "--n_hidden", type=int, help="Number of hidden units in LSTM layer.", default=50)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size.", default=64)
    parser.add_argument("-n", "--n_epoch", type=int, help="Number of epochs.", default=25)
    parser.add_argument("-n", "--split_seed", type=int, help="Seed for splitting the dataset.", default=44)
    parser.add_argument('-prep', "--preprocessing", type=bool, help="Preprocess dataset before training the model", default=True)

    args = parser.parse_args()
    os.mkdir(args.logdir)

    if args.use_pretrained:
        print('\nBuilding Embedding Matrix...')
        embedded_vocab_class = EmbeddedVocab(args.emb_path, args.emb_dim, args.download_emb, args.logdir+'embeddings/')
    else:
        embedded_vocab_class = None

    print('\nInitialized {embedded_vocab_class.embedding_size} dimensional embeddings for {len(embedded_vocab_class.reverse_vocab.keys())} words.')

    print('\nReading Dataset and splitting into train and test datasets with seed: ', args.split_seed)
    data = ImportData(args.data_file)
    data.train_test_split(seed=args.split_seed)

    print('\nPreprocessing Dataset...')
    train_dataset = QuoraQuestionDataset(data.train, pretrained_emb=args.use_pretrained, reverse_vocab=embedded_vocab_class, preprocess = args.preprocessing)
    train_dataset.words_to_ids()

    test_dataset = QuoraQuestionDataset(data.test, pretrained_emb=True, reverse_vocab=train_dataset.reverse_vocab, preprocess = args.preprocessing, train = False)
    test_dataset.words_to_ids()


    print('\n')
    print('Number of training samples        :', len(train_dataset))
    print('Number of validation samples      :', len(test_dataset))
    print('Number of unique words          :', train_dataset.unique_words)
    print('\n')


    print('\nBuilding model.')
    print('\nModel Parameters:')
    print('Hidden Size                  :', args.n_hidden)
    print('Batch Size                   :', args.batch_size)
    print('Number of Epochs             :', args.n_epoch)
    print('Use pretrained Embeddings    :', args.use_pretrained)
    print('Gradient clipping            :', args.gradient_clipping_norm)
    print('--------------------------------------\n')
    #pretrained embeddings

    n_hidden = args.n_hidden
    gradient_clipping_norm = args.gradient_clipping_norm
    batch_size = args.batch_size
    embeddings_dim = args.embeddings_dim
    n_epoch = args.n_epoch
    n_layer = args.n_layer
    n_token = train_dataset.unique_words
    use_pretrained_embeddings = args.use_pretrained
    train_emb = args.train_embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, collate_fn = generate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn = generate_batch)

    model = MaLSTM(n_hidden, embedded_vocab_class, embeddings_dim, n_layer, n_token, train_embeddings = train_emb, use_pretrained = use_pretrained_embeddings)
    model = model.float()
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    #średni czas liczenia epoki z CPU - ~1.8 h
    #Średni czas liczenia epoki z GPU (100 wymiarowe embeddingi) - ~4.8 minut
    #Średni czas liczenia epoki z GPU (300 wymiarowe embeddingi) - ~0.1238 h
    #Średni czas liczenia epoki z TPU - ?

    start = time()
    all_train_losses = []
    all_test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(n_epoch):
        epoch_time = time()

        
        epoch_iteration = 0
        
        epoch_loss=[]
        preds_train = []
        train(model, optimizer, criterion, train_dataloader, device)
          
        eval_loss = []
        preds_test = []
        eval(model, criterion, test_dataloader, device)
            
        train_loss = np.mean(epoch_loss)
        train_accuracy = np.sum(preds_train)/364348
        test_loss = np.mean(eval_loss)
        test_accuracy = np.sum(preds_test)/40000

        

        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'Mean losses of epoch {epoch} - train: {train_loss, train_accuracy}, test: {test_loss, test_accuracy}. Calculation time: {(time() - epoch_time)/3600} hours')
      
    print(time()-start)




