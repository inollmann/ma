import sys
import torch
import json
import csv
import compress_fasttext
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from jupy.transformer import Tokenizer


state_file = "d64_el1_dl1_ff64-ep1000.pt"
token_file = "dgs_vocab.json"
encoder_csv = "jupy/eval/encoder_pca.csv"
decoder_csv = "jupy/eval/decoder_pca.csv"
embedder_file = "models/word_embedding/cc.de.100.reduced.bin"
encoder_pt = "jupy/eval/encoder_embeddings.pt"
decoder_pt = "jupy/eval/decoder_embeddings.pt"
n_components = 64


save_pt = True
write_to_csv = False


state_dict = torch.load(state_file, map_location='cpu')

# Load Embeddings
try:
    encoder_weight = state_dict['src_embedding.weight']
    decoder_weight = state_dict['tgt_embedding.weight']
except KeyError:
    print("Could not find embedding keys in the state dict.")
    sys.exit(1)
pre_emb = compress_fasttext.models.CompressedFastTextKeyedVectors.load(embedder_file)

# Load Encoder Tokens
t = Tokenizer()
de_token_list = t.encoder.convert_ids_to_tokens([i for i in range(t.vocab_size)])
print("DE Vocabulary:", len(de_token_list))

# Load Decoder Tokens
with open('dgs_vocab.json', 'r', encoding="utf-8") as f:
    vocab = json.load(f)
dgs_idx2token = vocab["idx2token"]
print("DGS Vocabulary:", len(dgs_idx2token.values()))

# Load Pretrained Embeddings
# Encoder
first_iter = True
for token in de_token_list:
    if first_iter:
        encoder_pre_np = pre_emb[token]
        first_iter = False
    else:
        encoder_pre_np = np.vstack((encoder_pre_np, pre_emb[token]))
print("Original DE Embeddings:", encoder_pre_np.shape)
# Decoder
first_iter = True
for token in dgs_idx2token.values():
    if first_iter:
        decoder_pre_np = pre_emb[token]
        first_iter = False
    else:
        decoder_pre_np = np.vstack((decoder_pre_np, pre_emb[token]))
print("Original DGS Embeddings:", decoder_pre_np.shape)

# PCA
# Encoder Model Embeddings
encoder_np = encoder_weight.cpu().numpy()
pca_encoder = PCA(n_components=n_components)
encoder_pca = pca_encoder.fit_transform(encoder_np)
print("Model Encoder PCA:", encoder_pca.shape)
print(pca_encoder.explained_variance_ratio_)
print("Total variance retained:", pca_encoder.explained_variance_ratio_.sum(), "\n")
# Encoder Pretrained Embeddings
pca_encoder_pre = PCA(n_components=n_components)
encoder_pre_pca = pca_encoder_pre.fit_transform(encoder_pre_np)
print("Pretrained Encoder PCA:", encoder_pre_pca.shape)
print(pca_encoder_pre.explained_variance_ratio_)
print("Total variance retained:", pca_encoder_pre.explained_variance_ratio_.sum(), "\n")
# Decoder Model Embeddings
decoder_np = decoder_weight.cpu().numpy()
pca_decoder = PCA(n_components=n_components)
decoder_pca = pca_decoder.fit_transform(decoder_np)
print("Model Decoder PCA:", decoder_pca.shape)
print(pca_decoder.explained_variance_ratio_)
print("Total variance retained:", pca_decoder.explained_variance_ratio_.sum(), "\n")
# Decoder Pretrained Embeddings
pca_decoder_pre = PCA(n_components=n_components)
decoder_pre_pca = pca_decoder_pre.fit_transform(decoder_pre_np)
print("Pretrained Decoder PCA:", decoder_pre_pca.shape)
print(pca_decoder_pre.explained_variance_ratio_)
print("Total variance retained:", pca_decoder_pre.explained_variance_ratio_.sum(), "\n")

# Save PCA Embedding Tensor for Training
if n_components == 64 and save_pt:
    torch.save(torch.tensor(encoder_pre_pca), encoder_pt)
    torch.save(torch.tensor(decoder_pre_pca), decoder_pt)
    print("Saved PCA Embedding Tensors")

# Save PCA as CSV
if write_to_csv:
    # Log Encoder PCA
    encf = open(encoder_csv, "w", newline="", encoding='utf-8')
    encwriter = csv.writer(encf, delimiter=";")
    encwriter.writerow(["idx", "token", "mpca1", "mpca2", "ppca1", "ppca2"])

    for idx, (values, pvalues) in enumerate(zip(encoder_pca, encoder_pre_pca)):
        row = [idx, de_token_list[idx], values[0], values[1], pvalues[0], pvalues[1]]
        encwriter.writerow(row)
    encf.close()

    # Log Decoder PCA
    decf = open(decoder_csv, "w", newline="", encoding='utf-8')
    decwriter = csv.writer(decf, delimiter=";")
    decwriter.writerow(["idx", "token", "mpca1", "mpca2", "ppca1", "ppca2"])

    for (idx, token), values, pvalues in zip(dgs_idx2token.items(), decoder_pca, decoder_pre_pca):
        row = [idx, token, values[0], values[1], pvalues[0], pvalues[1]]
        decwriter.writerow(row)
    decf.close()

    print(f"Saved encoder PCA embeddings to {encoder_csv}")
    print(f"Saved decoder PCA embeddings to {decoder_csv}")

