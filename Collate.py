from torch.nn.utils.rnn import pad_sequence

''' Used to pad the dataset to have emails be equal length of words
'''
class Collate:
    ''' Initalizes the class
    '''
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    ''' perform the padding
    @:param
    batch: list of tuples from __getitem__
    '''
    def __call__(self, batch):
        # separate source and target into lists
        sources = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # pad them
        sources_pad = pad_sequence(sources, batch_first=True, padding_value=self.pad_idx)
        targets_pad = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return sources_pad, targets_pad