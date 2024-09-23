import torch
import torch.nn as nn
import torch.nn.functional as F
import compress_fasttext
import json
import pickle
import re
import numpy as np
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from torchtext.data import get_tokenizer


def load_vocabulary(vocab):
    return {idx: word for idx, word in enumerate(vocab)}, {word: idx for idx, word in enumerate(vocab)}


def load_vectors(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def tokenize(sentence):
    tok = get_tokenizer("moses", language='de')
    return tok(sentence)


def clean_token(token):
    if not re.search(r'[a-zA-Z]', token):
        token = ' '

    pattern = re.compile(r':(\d+)[\w\*]*')
    to_remove = ["$GEST-NM-", "$GEST-", "$NUM-", "$EXTRA-LING-"]
    number = ''

    if "$NUM" in token:
        match = pattern.search(token)
        if match:
            number = match.group(1)

    for substring in to_remove:
        token = token.replace(substring, '')

    token = token.lower()
    token = re.sub(r'\d.*', '', token)
    token = re.sub(r'[/^*|:$]', '', token)
    token = re.sub(r'-', ' ', token)
    token = token + number

    return token


class Seq2Seq(nn.Module):
    def __init__(self, vocabulary, max_iter=40, hidden_size=128):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = compress_fasttext.models.CompressedFastTextKeyedVectors.load('models/word_embedding/cc.de.300.reduced.bin')
        self.max_iter = max_iter
        self.idx2token, self.token2idx = load_vocabulary(vocabulary)
        self.idx2vec = self.generate_vocab_vectors()
        self.embedding_dim = self.idx2vec.size(dim=1)
        self.hidden_size = hidden_size
        self.sos = torch.tensor([0], device=self.device)
        self.eos = torch.tensor([1], device=self.device)

        self.encoder = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=2, batch_first=True).to(self.device)
        self.decoder = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=2, batch_first=True).to(self.device)
        self.dense = nn.Linear(self.hidden_size, len(self.idx2token)).to(self.device)
        self.softmax = nn.LogSoftmax(dim=-1).to(self.device)


    def forward(self, x, target_tensor=None):

        h0 = torch.zeros(self.encoder.num_layers, self.encoder.hidden_size, device=self.device)
        c0 = torch.zeros(self.encoder.num_layers, self.encoder.hidden_size, device=self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        _, (h, c) = self.encoder(x, (h0, c0))
        x = self.sos
        sentence = self.sos
        out = torch.empty((self.max_iter, len(self.idx2token)), dtype=torch.float, device=self.device)

        for i in range(self.max_iter):

            x = F.embedding(x, self.idx2vec)
            x, (h, c) = self.decoder(x, (h, c))
            x = self.dense(x)
            x = self.softmax(x)
            out[i] = x

            x = torch.argmax(x, dim=-1)

            sentence = torch.cat((sentence, x), -1)

            if target_tensor is not None:
                x = target_tensor[i].unsqueeze(0)

            if torch.equal(x, self.eos):   # <EOS> token
                out = out[:i+1]
                break

        return sentence[1:].unsqueeze(0), out


    def sentence2vec(self, sentence):
        tokens = tokenize(sentence)
        # print("[sentence2vec]", tokens)
        embeddings = [torch.tensor(self.embedder[token], device=self.device) for token in tokens]

        return torch.stack(embeddings).squeeze().to(self.device)


    def decode_sequence(self, sequence):
        sentence = []
        for v in sequence.squeeze():
            sentence.append(self.idx2token[v.item()])

        return sentence


    def encode_dgs(self, sentence):
        sentence = sentence.split()

        while "||" in sentence:
            sentence.remove("||")
        if not sentence:
            sentence = ["||"]

        tokens = []

        # for token in sentence:
        #     token = clean_token(token)
        #     tokens.append(token)

        encoded_tokens = torch.tensor([0], device=self.device)      # <SOS> token

        target = torch.zeros((self.max_iter, len(self.idx2token)), dtype=torch.long, device=self.device)

        for i, token in enumerate(sentence):
            if token in self.token2idx.keys():
                idx = torch.tensor(self.token2idx[token], device=self.device).unsqueeze(0)
            else:
                embeddings = torch.tensor(self.embedder[clean_token(token)], device=self.device).unsqueeze(0)
                similarities = F.cosine_similarity(embeddings, self.idx2vec, dim=-1)
                idx = similarities.argmax().unsqueeze(0)
            encoded_tokens = torch.cat((encoded_tokens, idx), dim=-1)
            target[i, idx] = 1

        encoded_tokens = torch.cat((encoded_tokens, torch.tensor([1], device=self.device)))     # <EOS> token
        encoded_tokens = encoded_tokens[1:]

        return encoded_tokens.to(self.device), target


    def generate_vocab_vectors(self):
        emb = self.embedder
        vec = []

        to_remove = ["$GEST-", "$NUM-", "$EXTRA-LING", "$GEST-NM-"]

        for idx, token in self.idx2token.items():
            idx = int(idx)
            if idx <= 3:
                vec.append(emb[token])
                continue

            token = clean_token(token)

            vec.append(emb[token])

        return torch.tensor(np.array(vec), device=self.device)
