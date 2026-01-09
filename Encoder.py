import torch
import torch.nn as nn

''' The Encoder Class compresses input data into context vectors to 
be used for summary of email into shorter pieces of text
'''
class Encoder(nn.Module):
    ''' The constructor method to define the main components of the model
    @Param
    input_size: the size of the working languages in n words
    embedding_size: the size of the context vector created
    hidden_size: number of neurons in the hidden LSTM layer
    layers: number of LSTM layers stacked together
    dropout: probability a neuron is turned off during training
    '''
    def __init__(self, input_size, embedding_size, hidden_size, layers, dropout):
        super().__init__()

        # Builds a matrix as a lookup table for words
        self.embedding = nn.Embedding(input_size, embedding_size)

        # Initializes an LSTM by creating weights and biases needed.
        # batch_first=True: setting change for data format
        self.rnn = nn.LSTM(embedding_size, hidden_size, layers, dropout=dropout, batch_first=True)

        # a dropout module to assist with overfitting
        self.dropout = nn.Dropout(dropout)

    ''' The method for execution of model
    @Param
    src: the input data (email batch) as a matrix of integers
    '''
    def forward(self, src):
        # Look up each vector and apply dropout noise
        embedded = self.dropout(self.embedding(src))

        # Run the LSTM
        # outputs: hidden state at each step for each word
        # hidden: current context
        # cell: overall context
        outputs, (hidden, cell) = self.rnn(embedded)

        # return the summarized vector
        return hidden, cell
