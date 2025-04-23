# # import mediapipe as mp
# import numpy as np
# import os
# import pickle
# import json
# import csv

# #%%
# # vocab_list = []
# # for idx, file in enumerate(os.listdir('vocab/vid/gebaerdenlernen')):
# #     if file.lower().endswith('.mp4'):
# #         name = file[:-4]
# #         vocab_list.append(name)
# #     else:
# #         continue
# vocab_list = [
#     "ich",
#     "du",
#     "er",
#     "sie",
#     "gehen",
#     "fahren",
#     "mögen",
#     "können",
#     "lesen",
#     "essen",
#     "haben",
#     "schlafen",
#     "trinken",
#     "Brot",
#     "Apfel",
#     "Salat",
#     "Auto",
#     "Fahrrad",
#     "Wasser",
#     "Kaffee",
#     "Milch"
# ]
# with open('vocab/words/subset.json', 'w') as outfile:
#     json.dump(vocab_list, outfile)
#
# #%%
# import compress_fasttext
# import json
# with open('vocab/words/subset.json', 'r') as f:
#     voc = json.load(f)
#
# model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('./models/word_embedding/cc.de.300.reduced.bin')
# dict = {idx: word for idx, word in enumerate(voc)}
# vecdict = {idx: model[word] for idx, word in enumerate(voc)}
#
# #%%
# print(len(dict))

#%%
# import torch
# from seq2seq_local import Seq2Seq
#
# model = Seq2Seq()
# dgsidx = model.encode_dgs("er trinkt Mineralwasser")
# test_input = model.sentence2vec("Fischers Fritz fischt frische Fische.")
# test_output = model(test_input)

#%%
# test_input = "Du isst gerne Brot."
# inp = model.sentence2vec(test_input)
# print(inp)
# test_output = model(inp)
# test_decoded_output = model.decode_sequence(test_output)
# print(test_input)
# print(test_output)
# print(test_decoded_output)
#
# #%%
# import torch
# import tensorflow as tf
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter('./runs')
#
# #%%
# writer.add_graph(model, inp)
# writer.close()

#%%
# import re
#
# with open('./vocab/translations/translations_subset.txt', encoding='utf-8') as f:
#     data = f.read().splitlines()
# pairs = [re.split(r'\s{2,}', line) for line in data]

#%%
# from torch.utils.data import DataLoader
# from dataset import DgsDataset

# ds = DgsDataset('./vocab/translations/translations_subset.txt')
# dl = DataLoader(ds, batch_size=1, shuffle=True)

# de, dgs, de_string, dgs_string = next(iter(dl))
# print(de, de_string)
# print(dgs, dgs_string)

#%%
# import csv
#
# with open('vocab/translations/dgs_korpus.CSV') as f:
#     csv_file = csv.reader(f, delimiter=';')
#     data = {'id': [], 'de': [], 'dgs': [], 'mouth': []}
#     cur_idx, cur_de, cur_dgs, cur_mouth = None, "", "", ""
#
#     for row in csv_file:
#         if row[0] and row[0] != ' ':
#             print(row[0])
#             cur_idx = int(row[0])
#
#         if row[1] == 'Deutsch':
#             cur_de = row[2]
#         elif row[1] == 'Gloss':
#             cur_dgs = row[2]
#         elif row[1] == 'Mundbild':
#             cur_mouth = row[2]
#
#         if not any(row) and any([cur_idx, cur_de, cur_dgs, cur_mouth]):
#             data['id'].append(cur_idx)
#             data['de'].append(cur_de)
#             data['dgs'].append(cur_dgs)
#             data['mouth'].append(cur_mouth)
#             cur_idx, cur_de, cur_dgs, cur_mouth = None, "", "", ""
#
# #%%
# for id, de, dgs, mouth in zip(data['id'], data['de'], data['dgs'], data['mouth']):
#     print(id, de, dgs, mouth)
#
# import pickle
# pickle.dump(data, open('vocab/translations/dgs_korpus.pkl', 'wb'))

#%%
# from collections import Counter
# import pickle
# import itertools
# from torchtext.data import get_tokenizer
#
# with open('vocab/translations/dgs_korpus.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# sentences = [entry.split() for entry in data['dgs']]
# tokens = list(itertools.chain.from_iterable(sentences))
#
# keys = Counter(tokens).keys()
# counts = Counter(tokens).values()
# c = Counter(tokens)
#
# vocabulary = sorted(c, key=c.get, reverse=True)
# tok = get_tokenizer("moses", language='de')
#
# #%%
# print(tok(vocabulary[56]))

#%%
# import pickle
#
# with open('vocab/translations/dgs_korpus.pkl', 'rb') as f:
#     data = pickle.load(f)
# index = 2678
# test = data['de'][index]
# print(data['de'][index])
# print(data['dgs'][index])
# del data['de'][index]
# del data['dgs'][index]
# del data['id'][index]
# del data['mouth'][index]
# print(data['de'][index])
# #%%
# pickle.dump(data, open('vocab/translations/dgs_korpus.pkl', 'wb'))

#%%
# from dataset import DgsDataset

# ds = DgsDataset('vocab/translations/dgs_korpus.pkl')
# reduced = ds.reduced_vocabulary

#%%
# import pickle
# with open('vocab/translations/from_transcripts.pkl', 'rb') as f:
#     data = pickle.load(f)
# pass

# %%
# import csv

# with open('dataset.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     fieldnames = list(data.keys())
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     for i in range(len(data["transcript"])):
#         writer.writerow({field: data[field][i] for field in fieldnames})

# %%
# from transformer import Tokenizer

# german_tokenizer = Tokenizer()
# print("Tokenizing...")
#%%
# v = german_tokenizer.encode("Zehn zahme Ziegen zogen zehn Zentner Zucker zum Zoo.")
# print(v)
# print(german_tokenizer.encoder.all_special_ids)
# print(german_tokenizer.encoder.special_tokens_map)
# print(german_tokenizer.encoder.decode(v))

# %%
# %% [code]
import math
import copy
import random
import ast
import re
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import spacy
from transformers import AutoTokenizer

# -----------------------------
# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    print("cuda")
    
# -----------------------------
# Define Special Tokens (using square brackets consistently)
PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
NAME_TOKEN = "[NAME]"
NUM_TOKEN = "[NUM]"
UNK_TOKEN = "[UNK]"

special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, NAME_TOKEN, NUM_TOKEN, UNK_TOKEN]

# -----------------------------
# 1. Data Loading, DGS Preprocessing, and Vocabulary Building

# Load dataset
file_path = "dataset.csv"
df = pd.read_csv(file_path)
print("Dataset shape:", df.shape)
print(df.head())

# Convert the DGS column (assumed to be stored as a string representation of a list) to an actual list
df['dgs'] = df['dgs'].apply(lambda x: ast.literal_eval(x))
print("\nSample DGS tokens:", df['dgs'].iloc[0])

# Define a cleaning function for DGS tokens
def clean_token(token):
    # Replace tokens with special tokens if they contain "$NUM" or "$NAME"
    if "$NUM" in token:
        return NUM_TOKEN
    if "$NAME" in token:
        return NAME_TOKEN

    # If no alphabetic characters, return a space.
    if not re.search(r'[a-zA-Z]', token):
        token = ' '

    pattern = re.compile(r':(\d+)[\w\*]*')
    to_remove = ["$GEST-NM-", "$GEST-", "$NUM-", "$EXTRA-LING-"]
    number = ''
    if "$NUM" in token or "$INDEX" in token:
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

# Clean DGS tokens
df['dgs'] = df['dgs'].apply(lambda tokens: [clean_token(token) for token in tokens])

# Filter: Keep only rows where DGS tokens length <= max_tokens, and filter out any tokens containing "$LIST" or "$ALPHA"
max_tokens = 10
df = df[df['dgs'].apply(lambda tokens: len(tokens) <= max_tokens)].reset_index(drop=True)
df = df[~df['dgs'].apply(lambda tokens: any(("$LIST" in token or "$ALPHA" in token) for token in tokens))].reset_index(drop=True)

print("DGS vocab size (before building):", df['dgs'].apply(len).sum())  # rough info
print("Elements in dataset:", len(df["transcript"]))

# Build vocabulary from the DGS tokens
def build_vocab(token_lists, min_freq=1):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)
    # Keep tokens with frequency >= min_freq
    vocab = {tok for tok, cnt in counter.items() if cnt >= min_freq}
    vocab = sorted(list(vocab))
    # Prepend our special tokens
    vocab = special_tokens + vocab
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token

dgs_token2idx, dgs_idx2token = build_vocab(df['dgs'].tolist())
print("Final DGS vocabulary size:", len(dgs_token2idx))

# Optionally, save the DGS vocabulary
import json
dgs_vocab = {"token2idx": dgs_token2idx, "idx2token": dgs_idx2token}
with open("dgs_vocab.json", 'w', encoding="utf-8") as f:
    json.dump(dgs_vocab, f)

# -----------------------------
# 2. Define a Custom German Tokenizer Class
class Tokenizer():
    def __init__(self, extra_tokens=['[EOS]', '[NUM]', '[NAME]']):
        # Load spaCy German model (using md for better NER)
        self.nlp = spacy.load("de_core_news_md")
        # Load the Hugging Face tokenizer (for German)
        self.encoder = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        # Add extra special tokens
        if extra_tokens:
            extra_token_dict = {'additional_special_tokens': extra_tokens}
            self.num_extra_tokens = self.encoder.add_special_tokens(extra_token_dict)
        else:
            self.num_extra_tokens = 0
        # Update our vocab_size from the updated tokenizer
        self.vocab_size = len(self.encoder)
        # Use the correct pad_token_id
        self.pad_token_id = self.encoder.pad_token_id
        # Define a list of German number words for extra matching
        self.__numcheck = ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", 
                           "zehn", "zwölf", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig",
                           "siebzig", "achtzig", "neunzig", "hundert", "tausend", "illion", "illiarde"]

    def preprocess(self, src_text):
        """Use spaCy to detect person names and numbers and replace them with special tokens."""
        doc = self.nlp(src_text)
        new_tokens = []
        for token in doc:
            if token.ent_type_ == "PER":
                new_tokens.append("[NAME]")
            elif token.like_num or any(ch.isdigit() for ch in token.text) or any(sub in token.text.lower() for sub in self.__numcheck):
                new_tokens.append("[NUM]")
            else:
                new_tokens.append(token.text)
        # Optionally, further clean the sentence (remove extraneous punctuation)
        sentence = " ".join(new_tokens)
        sentence = re.sub(r'[\/^*|:!?.]', '', sentence)
        return sentence

    def encode(self, src_text, max_length=32):
        # Preprocess the text
        preprocessed_text = self.preprocess(src_text)
        # Encode using the Hugging Face tokenizer (do not add its own special tokens)
        token_ids = self.encoder.encode(preprocessed_text, add_special_tokens=False, 
                                          max_length=max_length, truncation=True)
        # Append the EOS token ID
        eos_id = self.encoder.convert_tokens_to_ids("[EOS]")
        token_ids.append(eos_id)
        return token_ids

# Instantiate our German tokenizer
de_tokenizer = Tokenizer()
print("German tokenizer vocab size:", de_tokenizer.vocab_size)

# -----------------------------
# 3. Define the PyTorch Dataset and Collate Function
class TranslationDataset(Dataset):
    def __init__(self, df, de_tokenizer, dgs_token2idx, max_src_len=32, max_tgt_len=32):
        self.df = df.reset_index(drop=True)
        self.de_tokenizer = de_tokenizer
        self.dgs_token2idx = dgs_token2idx
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # German sentence tokenization using our custom tokenizer
        src_text = self.df.loc[idx, "de"]
        src_tokens = self.de_tokenizer.encode(src_text, max_length=self.max_src_len)
        # DGS tokens: add SOS and EOS
        tgt_tokens = self.df.loc[idx, "dgs"]
        tgt_tokens = [SOS_TOKEN] + tgt_tokens + [EOS_TOKEN]
        # Convert DGS tokens to indices (use UNK if not found)
        tgt_indices = [self.dgs_token2idx.get(tok, self.dgs_token2idx[UNK_TOKEN]) for tok in tgt_tokens]
        tgt_indices = tgt_indices[:self.max_tgt_len]
        sample = {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_indices, dtype=torch.long)
        }
        return sample

def collate_fn(batch):
    src_seqs = [b["src"] for b in batch]
    tgt_seqs = [b["tgt"] for b in batch]
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, padding_value=de_tokenizer.pad_token_id, batch_first=True)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, padding_value=dgs_token2idx[PAD_TOKEN], batch_first=True)
    return {"src": src_padded, "tgt": tgt_padded}

train_df, val_df = train_test_split(df, test_size=0.05, random_state=SEED)
train_dataset = TranslationDataset(train_df, de_tokenizer, dgs_token2idx)
val_dataset = TranslationDataset(val_df, de_tokenizer, dgs_token2idx)
BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# 4. Define the Transformer Model Components

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, model_dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if src_mask is not None and src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(dtype=src_mask.dtype)
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                          key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(attn_output))
        ff_output = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + self.dropout(ff_output))
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if tgt_mask is not None and tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(dtype=tgt_mask.dtype)
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        if memory_mask is not None and memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.to(dtype=memory_mask.dtype)
        enc_dec_attn_output, _ = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(enc_dec_attn_output))
        ff_output = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + self.dropout(ff_output))
        return tgt

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
    def forward(self, src, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
        return tgt

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.src_embedding = nn.Embedding(src_vocab_size, model_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        self.pos_decoder = PositionalEncoding(model_dim, dropout)
        encoder_layer = TransformerEncoderLayer(model_dim, num_heads, ff_dim, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(model_dim, num_heads, ff_dim, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(model_dim, tgt_vocab_size)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if tgt_mask is not None and tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(dtype=tgt_mask.dtype)
        src = self.src_embedding(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.model_dim)
        tgt = self.pos_decoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        output = self.fc_out(output)
        return output

# -----------------------------
# 5. Define Training and Evaluation Functions with TensorBoard Logging

def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    src_key_padding_mask = (src == src_pad_idx)
    tgt_key_padding_mask = (tgt == tgt_pad_idx)
    return src_key_padding_mask, tgt_key_padding_mask

def train_epoch(model, dataloader, optimizer, criterion, src_pad_idx, tgt_pad_idx, device, writer, epoch):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        src = batch["src"].to(device)  # (batch_size, src_seq_len)
        tgt = batch["tgt"].to(device)  # (batch_size, tgt_seq_len)
        optimizer.zero_grad()
        src = src.transpose(0, 1)  # (src_seq_len, batch_size)
        tgt = tgt.transpose(0, 1)  # (tgt_seq_len, batch_size)
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :].contiguous().view(-1)
        src_kpm, tgt_kpm_full = create_masks(src.transpose(0,1), tgt.transpose(0,1), src_pad_idx, tgt_pad_idx)
        tgt_kpm = tgt_kpm_full[:, :-1]
        tgt_seq_len = tgt_input.size(0)
        tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=src.device) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        output = model(src, tgt_input, src_mask=None, tgt_mask=tgt_mask,
                       src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm)
        output = output.view(-1, output.shape[-1])
        loss = criterion(output, tgt_out)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(dataloader) + i)
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, src_pad_idx, tgt_pad_idx, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :].contiguous().view(-1)
            src_kpm, tgt_kpm_full = create_masks(src.transpose(0,1), tgt.transpose(0,1), src_pad_idx, tgt_pad_idx)
            tgt_kpm = tgt_kpm_full[:, :-1]
            tgt_seq_len = tgt_input.size(0)
            tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=src.device) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
            output = model(src, tgt_input, src_mask=None, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm)
            output = output.view(-1, output.shape[-1])
            loss = criterion(output, tgt_out)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# -----------------------------
# 6. Instantiate Model, Optimizer, Loss, and TensorBoard Writer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# For the German tokenizer, use its updated vocab_size:
SRC_VOCAB_SIZE = de_tokenizer.vocab_size  # Already updated in our Tokenizer class
TGT_VOCAB_SIZE = len(dgs_token2idx)  # Our DGS vocabulary size

MODEL_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FF_DIM = 1024
DROPOUT = 0.1

model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                    num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
                    ff_dim=FF_DIM, dropout=DROPOUT).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total learnable parameters: {total_params}")

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=dgs_token2idx[PAD_TOKEN])

writer = SummaryWriter(log_dir="runs/transformer")

# Log model graph with dummy inputs
sample_src = torch.randint(0, SRC_VOCAB_SIZE, (35, 1), dtype=torch.long, device=device)
sample_tgt = torch.randint(0, TGT_VOCAB_SIZE, (20, 1), dtype=torch.long, device=device)
dummy_src_mask = torch.zeros((sample_src.size(0), sample_src.size(0)), device=device)
dummy_tgt_mask = torch.zeros((sample_tgt.size(0), sample_tgt.size(0)), device=device)
dummy_memory_mask = torch.zeros((sample_tgt.size(0), sample_src.size(0)), device=device)
dummy_src_key_padding_mask = torch.zeros((sample_src.size(1), sample_src.size(0)), dtype=torch.bool, device=device)
dummy_tgt_key_padding_mask = torch.zeros((sample_tgt.size(1), sample_tgt.size(0)), dtype=torch.bool, device=device)
dummy_memory_key_padding_mask = torch.zeros((sample_src.size(1), sample_src.size(0)), dtype=torch.bool, device=device)
writer.add_graph(model, (sample_src, sample_tgt, 
                           dummy_src_mask, dummy_tgt_mask, dummy_memory_mask, 
                           dummy_src_key_padding_mask, dummy_tgt_key_padding_mask, dummy_memory_key_padding_mask))

# -----------------------------
# 7. Training Loop
NUM_EPOCHS = 2
for epoch in range(1, NUM_EPOCHS+1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion,
                             src_pad_idx=de_tokenizer.pad_token_id,
                             tgt_pad_idx=dgs_token2idx[PAD_TOKEN], device=device, 
                             writer=writer, epoch=epoch)
    val_loss = evaluate(model, val_loader, criterion,
                        src_pad_idx=de_tokenizer.pad_token_id,
                        tgt_pad_idx=dgs_token2idx[PAD_TOKEN], device=device)
    writer.add_scalar("Loss/train_epoch", train_loss, epoch)
    writer.add_scalar("Loss/val_epoch", val_loss, epoch)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
    writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
    torch.save(model.state_dict(), "transformer_v2.pt")
writer.close()

# -----------------------------
# 8. Greedy Decoding Function for Inference
def greedy_decode(model, src_sentence, max_len, de_tokenizer, dgs_token2idx, dgs_idx2token, device):
    model.eval()
    # Tokenize source sentence using our custom tokenizer (which returns token IDs)
    src_tokens = de_tokenizer.encode(src_sentence, max_length=32)
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, src_seq_len)
    src_tensor = src_tensor.transpose(0,1)  # (src_seq_len, 1)
    # Obtain encoder memory (without masks for simplicity)
    memory = model.encoder(model.pos_encoder(model.src_embedding(src_tensor) * math.sqrt(model.model_dim)))
    tgt_indices = [dgs_token2idx[SOS_TOKEN]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(1).to(device)
        tgt_tensor = model.pos_decoder(model.tgt_embedding(tgt_tensor) * math.sqrt(model.model_dim))
        tgt_mask = torch.triu(torch.ones((tgt_tensor.size(0), tgt_tensor.size(0)), device=device) == 1).transpose(0,1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        out = model.decoder(tgt_tensor, memory, tgt_mask=tgt_mask)
        out = model.fc_out(out)
        prob = out[-1, 0]
        next_token = torch.argmax(prob).item()
        tgt_indices.append(next_token)
        if next_token == dgs_token2idx[EOS_TOKEN]:
            break
    decoded_tokens = [dgs_idx2token[idx] for idx in tgt_indices if idx not in {dgs_token2idx[SOS_TOKEN], dgs_token2idx[EOS_TOKEN]}]
    return decoded_tokens

# Example inference:
example_sentence = "Ich glaube, das Netzwerk ist gut genug trainiert."
decoded_dgs = greedy_decode(model, example_sentence, max_len=20,
                            de_tokenizer=de_tokenizer,
                            dgs_token2idx=dgs_token2idx, dgs_idx2token=dgs_idx2token, device=device)
print("Input German sentence:", example_sentence)
print("Decoded DGS tokens:", decoded_dgs)
