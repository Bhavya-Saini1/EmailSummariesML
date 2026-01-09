import torch
import torch.nn as nn
import random

''' The Seq2Seq classes utilizes the encoder and decoder 
together and uses teacher forcing to help train the model
'''
class Seq2Seq(nn.Module):
    '''
    @Param
    encoder: the encoder model
    decoder: the decoder model
    device: device used for training
    '''
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    '''
    @Param
    src: an email
    trg: the summary
    teacher_forcing_ratio: how often teacher forcing is used
    '''
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]

        trg_len = trg.shape[1]

        # get size of summary output
        trg_vocab_size = self.decoder.output_size

        # Will store outputs of the decoder
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # encode the original text
        hidden, cell = self.encoder(src)

        # input of decoder, the token
        input = trg[:, 0].unsqueeze(1)

        # step by step decoding of vector
        for t in range(1, trg_len):
            # decode a step
            output, hidden, cell = self.decoder(input, hidden, cell)

            # store the prediction
            outputs[:, t] = output

            # decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # or the actual word
            top1 = output.argmax(1)

            # assign it to next input
            input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        # return the predictions
        return outputs
