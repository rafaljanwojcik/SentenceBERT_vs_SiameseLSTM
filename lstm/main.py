import argparse

import torch

from data import ImportData, QuoraQuestionDataset
from embeddings import EmbeddedVocab
from model import MaLSTM
from utils import generate_batch, train, eval

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--logdir", type=str, help="Directory to save all downloaded files, and model checkpoints.", default='/logs/')  
    parser.add_argument("-df", "--data_file", type=str, help="Path to dataset.", default="../logs/questions.csv")
    parser.add_argument("-e", "--use_pretrained", type=bool, help="Boolean, whether use pretrained embeddings.", default=False)
    parser.add_argument("-e", "--emb_filename", type=str, help="Name of file with embeddings.", default='glove.6B.100d.txt')
    parser.add_argument("-e", "--emb_path", type=str, help="path to file with embeddings", default='glove.6B.100d.txt')
    parser.add_argument("-z", "--n_hidden", type=int, help="Number of hidden units in LSTM layer.", default=50)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size.", default=64)
    parser.add_argument("-n", "--n_epoch", type=int, help="Number of epochs.", default=25)
    parser.add_argument("-n", "--split_seed", type=int, help="Seed for splitting the dataset.", default=44)
    parser.add_argument('-prep', "--preprocessing", type=bool, help="Preprocess dataset before training the model", default=True)

    args = parser.parse_args()

    if args.use_pretrained:
        print('Building Embedding Matrix...')
        embedded_vocab_class = EmbeddedVocab(args.emb_file_name, args.embedding_path, args.embeddings_dim, args.download_emb, args.logdir)
    else:
        embedded_vocab_class = None

    print('\nInitialized {embedded_vocab_class.embedding_size} dimensional embeddings for {len(embedded_vocab_class.reverse_vocab.keys())} words.')

    print('\nReading Dataset and splitting into train and test datasets with seed: ', args.split_seed)
    data = ImportData(args.data_file)
    data.train_test_split(seed=args.split_seed)

    print('\nPreprocessing Dataset...')
    train_dataset = QuoraQuestionDataset(data.train, pretrained_emb=args.use_pretrained, reverse_vocab=embedded_vocab_class)
    train_dataset.preprocessing(cleaning=args.preprocessing)

    test_dataset = QuoraQuestionDataset(data.test, pretrained_emb=True, reverse_vocab=embedded_vocab_class)
    test_dataset.preprocessing(cleaning=args.preprocessing)


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
    pretrained_embeddings = args.use_pretrained
    train_emb = args.train_embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, collate_fn = generate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn = generate_batch)

    model = MaLSTM(n_hidden, embedded_vocab_class, embeddings_dim, n_layer, n_token, train_embeddings = train_emb, use_pretrained = pretrained_embeddings)
    model = model.float()
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adadelta(model.parameters())



