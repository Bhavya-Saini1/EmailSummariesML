# EmailSummariesML

A sequence-to-sequence LSTM that generates a subject line from the body of an email, written from scratch in PyTorch.

No pretrained transformer and no summarization API. The encoder, decoder, attention-free context passing, vocabulary, tokenizer pipeline, batching, and training loop are all implemented directly against `torch.nn` primitives. The goal was to understand how neural summarization works end to end rather than to call one.

## Task

Given an email body, produce a short subject line that captures its intent.

```
Input:   "please send the report by friday"
Output:  "send report"
```

## Data

The [AESLC](https://github.com/ryanzhumich/AESLC) corpus, a set of Enron emails paired with their human-written subject lines.

| Split | Examples |
|---|---|
| `train.csv` | 14,436 |
| `validation.csv` | 1,960 |
| `test.csv` | 1,906 |

Each row has an `email_body` and a `subject_line`. Bodies are truncated to 300 tokens and subject lines to 20, which covers the large majority of the corpus while keeping the unrolled decoder a manageable depth.

## Architecture

```
email body
    |
    v
Vocabulary          spaCy tokenizer, lowercased, min frequency 3
    |               <PAD> <SOS> <EOS> <UNK> reserved at indices 0-3
    v
Encoder             Embedding(vocab, 256)
    |               LSTM(256 -> 512, 2 layers, dropout 0.5)
    |
    +--> (hidden, cell) context
    |
    v
Decoder             Embedding(vocab, 256)
    |               LSTM(256 -> 512, 2 layers, dropout 0.5)
    |               Linear(512 -> vocab)
    v
subject line        greedy decode, stops at <EOS> or 20 tokens
```

The encoder compresses the whole body into a fixed `(hidden, cell)` pair. The decoder is then unrolled one token at a time from that state.

**Teacher forcing** is applied at a ratio of 0.5. At each decoding step the model either receives its own previous prediction or the ground-truth token, chosen at random. Without this, one early wrong token derails the rest of the sequence and the model struggles to learn long-range structure.

### Files

| File | Role |
|---|---|
| `Vocabulary.py` | spaCy tokenization, frequency-thresholded vocab, `numericalize` |
| `CustomDataset.py` | `torch.utils.data.Dataset` over the CSV, truncation, `<SOS>`/`<EOS>` wrapping |
| `Collate.py` | Pads variable-length sequences within a batch via `pad_sequence` |
| `Encoder.py` | Embedding plus stacked LSTM, returns the context state |
| `Decoder.py` | Single-step decode, returns logits over the vocabulary |
| `Seq2Seq.py` | Unrolls the decoder, applies teacher forcing |
| `training.py` | Training loop plus a `predict` helper for greedy inference |
| `save.py` | Checkpoint save and load |
| `Journal.txt` | Engineering journal, see below |

## Training setup

| Hyperparameter | Value |
|---|---|
| Embedding dimension | 256 |
| Hidden size | 512 |
| LSTM layers | 2 |
| Dropout | 0.5 |
| Batch size | 32 |
| Optimizer | Adam, learning rate 1e-3 |
| Loss | Cross entropy, ignoring `<PAD>` |
| Gradient clipping | Max norm 1.0 |

Loss ignores padding positions so that short sequences in a padded batch do not reward the model for predicting `<PAD>`. Gradients are clipped because the unrolled LSTM is prone to exploding gradients on longer bodies.

The device is selected automatically, preferring Apple Silicon MPS, then CUDA, then CPU. Checkpoints are written every 5 epochs.

## Results

On a small dummy dataset used to validate that the architecture learns at all, loss converged cleanly:

| Epoch | Loss |
|---|---|
| 10 | 1.3080 |
| 20 | 0.1624 |
| 50 | 0.0038 |
| 100 | 0.0011 |

This is memorization of a tiny sample, not a generalization result, and it is reported only as evidence that the gradient path is wired correctly end to end. Full-corpus evaluation on the AESLC validation and test splits is the next step, with ROUGE as the intended metric.

## Journal

`Journal.txt` is a running engineering log kept during the build. It records the design decisions and their reasoning, every reference consulted, and every error hit along with the diagnosis:

- Why a generative seq2seq model was chosen over extractive frequency ranking, since summarization is not a classification problem.
- A 4D tensor reaching the LSTM because of an extra `unsqueeze` in the decoder.
- Hidden and cell states arriving 3D against a 2D input during single-step inference, which does not surface during training because training runs batched.
- A `spacy` model resolving against the wrong virtualenv.

It is kept in the repository deliberately. The reasoning behind a model is usually harder to reconstruct later than the code.

## Running it

```bash
pip install torch pandas spacy
python -m spacy download en_core_web_sm

python training.py
```

`training.py` expects `train.csv` in the working directory.

## Known issues

- `Vocabulary.py` calls `spacy.cli.download` at import time, so the model is re-fetched on every run. It belongs in setup instructions instead.
- The vocabulary is rebuilt from scratch on each run rather than being serialized alongside the checkpoint, so a loaded checkpoint is only valid against an identically rebuilt vocab.
- There is no attention mechanism. A single fixed-width context vector is a real bottleneck for longer emails, and attention is the most valuable next addition.
- Evaluation is by inspection. There is no ROUGE scoring or validation loop yet.

## License

MIT, see [LICENSE](LICENSE).
