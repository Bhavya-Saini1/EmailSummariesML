import torch
from torch.utils.data import Dataset
import pandas as pd


''' Data reader and handler class
'''
class CustomDataset(Dataset):
    '''
    @:param
    df: The pandas dataframe containing data
    vocab: Your initialized Vocabulary object
    source_col: The name of the column with emails (e.g. "email_body")
    target_col: The name of the column with summaries (e.g. "subject_line")
    '''
    def __init__(self, df, vocab, source_col, target_col):
        self.df = df
        self.vocab = vocab
        self.source_col = source_col
        self.target_col = target_col

    ''' for getting length of the dataframe
    '''
    def __len__(self):
        return len(self.df)

    ''' for getting an item from the datset
    @:param
    index: The index of the datset to get
    '''
    def __getitem__(self, index):
        # get the text for row index
        source_text =  self.df.iloc[index][self.source_col]
        target_text = self.df.iloc[index][self.target_col]

        # turn them into numbers using vocab
        source_indices = self.vocab.numericalize(source_text)
        target_indices = self.vocab.numericalize(target_text)

        # add the SOS and EOS tokens
        source_list = [self.vocab.stoi["<SOS>"]] + source_indices + [self.vocab.stoi["<EOS>"]]
        target_list = [self.vocab.stoi["<SOS>"]] + target_indices + [self.vocab.stoi["<EOS>"]]

        source_tensor =  torch.tensor(source_list, dtype=torch.long)
        target_tensor =  torch.tensor(target_list, dtype=torch.long)

        return source_tensor, target_tensor