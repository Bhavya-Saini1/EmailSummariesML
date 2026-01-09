import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from CustomDataset import CustomDataset
from Collate import Collate
from Vocabulary import Vocabulary
from Encoder import Encoder
from Decoder import Decoder
from Seq2Seq import Seq2Seq

# hyperparameters
input_size = 0 # will update after vocab is built
output_size = 0 # will update after vocab is built
emb_dim = 256
hidden_size = 512
layers = 2
dropout = 0.5
batch_size = 32
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data Preparation

data = {
    'email': [
        'meeting at 5 pm regarding the project',
        'please send the report by friday',
        'can we reschedule our call to monday',
        'urgent update required for the client'
    ],
    'summary': [
        'meeting 5pm',
        'send report',
        'reschedule call',
        'client update'
    ]
}
df = pd.DataFrame(data)

# threshold can be 1 since we have tiny data
vocab = Vocabulary(freq_threshold=1)
all_text = df['email'].tolist() + df['summary'].tolist()
vocab.build_vocabulary(all_text)

# model Initialization
input_size = len(vocab)
output_size = len(vocab)

encoder = Encoder(input_size, emb_dim, hidden_size, layers, dropout).to(device)
decoder = Decoder(output_size, emb_dim, hidden_size, layers, dropout).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)


# dataLoader
dataset = CustomDataset(df, vocab, 'email', 'summary')
pad_idx = vocab.stoi["<PAD>"]
loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    collate_fn=Collate(pad_idx=pad_idx),
    shuffle=True
)

print("Setup done")


# optimizer updates weights to reduce error
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# loss Function calculates how wrong the model is
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# how many times to loop through the dataset
num_epochs = 100

for epoch in range(num_epochs):
    model.train()  # set to training mode
    epoch_loss = 0

    for batch_idx, (src, trg) in enumerate(loader):
        src = src.to(device)
        trg = trg.to(device)


        output = model(src, trg)

        # reshape for Loss Calculation
        output_dim = output.shape[-1]

        # flatten the data
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        # backpropagation
        optimizer.zero_grad()  # Clear old gradients
        loss = criterion(output, trg)  # Calculate error
        loss.backward()  # Calculate gradients

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        epoch_loss += loss.item()

    # prints average loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1} | Loss: {epoch_loss / len(loader):.4f}')

print("Training Done")