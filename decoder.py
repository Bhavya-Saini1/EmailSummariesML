import torch
import torch.nn as nn

''' The Decoder class generates a summary of the email given the context vector
'''
class Decoder(nn.Module):
    ''' The constructor method to define the main components of the model
    @Param
    output_size: the size of the final summary output
    embedding_size: the size of the context vector
    hidden_size: number of neurons in the hidden LSTM layer
    layers: number of LSTM layers stacked together
    dropout: probability a neuron is turned off during training
    '''
    def __init__(self, output_size, embedding_size, hidden_size, layers, dropout):
        super().__init__()

        # size of vocabulary used
        self.output_size = output_size

        # Builds a matrix as a lookup table for words
        self.embedding = nn.Embedding(output_size, embedding_size)

        # Initializes an LSTM by creating weights and biases needed
        self.rnn = nn.LSTM(embedding_size, hidden_size, layers, dropout=dropout, batch_first=True)

        # A linear layer to predict the next word given the hidden state
        self.fc_out = nn.Linear(hidden_size, output_size)

        # a dropout module to assist with overfitting
        self.dropout = nn.Dropout(dropout)

    ''' The method for execution of model
    @Param
    input: the token for the word previously generated
    hidden: Context vector from last step
    cell: Cell State from last step
    '''
    def forward(self, input, hidden, cell):
        # embedding expects dimension
        input = input.unsqueeze(1)

        # Look up each vector and apply dropout noise
        embedded = self.dropout(self.embedding_size(input))

        # Run the LSTM, given the previous hidden and cell states
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # make a prediction
        prediction = self.fc_out(output.squeeze(1))

        # return the new vector for next step
        return prediction, hidden, cell