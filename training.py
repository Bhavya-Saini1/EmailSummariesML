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


def predict(sentence):
    model.eval()
    with torch.no_grad():
        # process Input
        tokens = vocab.tokenizer_eng(sentence)
        indices = [vocab.stoi.get(t, vocab.stoi["<UNK>"]) for t in tokens]

        # add <SOS> and <EOS>
        indices = [vocab.stoi["<SOS>"]] + indices + [vocab.stoi["<EOS>"]]

        # convert to tensor and add batch dimension
        src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)

        # feed into Encoder
        hidden, cell = model.encoder(src_tensor)

        # build Summary Word by Word
        outputs = [vocab.stoi["<SOS>"]]

        for _ in range(20):  # Max length 20 words
            previous_word = torch.LongTensor([outputs[-1]]).unsqueeze(0).to(device)

            with torch.no_grad():
                output, hidden, cell = model.decoder(previous_word, hidden, cell)

                # Get the highest probability
                best_guess = output.argmax(1).item()

            outputs.append(best_guess)

            # stop if model predicts <EOS>
            if best_guess == vocab.stoi["<EOS>"]:
                break

        # convert indices back to words
        translated_sentence = [vocab.itos[idx] for idx in outputs]

        return " ".join(translated_sentence[1:-1])


# test with a sentence from training data
print("\nTESTING:")
test_sentence = "please send the report by friday"
print(f"Original: {test_sentence}")
print(f"Summary: {predict(test_sentence)}")