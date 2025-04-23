print("Loading transformer_model.py")

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import copy
import spacy
from transformers import AutoTokenizer


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
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, model_dim)
        """
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
        # Self-attention sublayer
        if src_mask is not None and src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(dtype=src_mask.dtype)

        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                          key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(attn_output))
        # Feed-forward sublayer
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
        # Convert tgt_key_padding_mask if needed for self-attention:
        if tgt_mask is not None and tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(dtype=tgt_mask.dtype)
        self_attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                               key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_output))
        
        # Convert memory_key_padding_mask if needed for encoder-decoder attention:
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

        # Source embedding (German tokens)
        self.src_embedding = nn.Embedding(src_vocab_size, model_dim)
        # Target embedding (DGS tokens)
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
        """
        Args:
            src: source sequence tensor (seq_len, batch_size)
            tgt: target sequence tensor (seq_len, batch_size)
        """
        # Embed and add positional encoding
        tgt_key_padding_mask = tgt_key_padding_mask.to(dtype=tgt_mask.dtype) ###
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


class Tokenizer():
    def __init__(self, extra_tokens=['[EOS]', '[NUM]', '[NAME]']):
        self.nlp = spacy.load("de_core_news_md")
        self.encoder = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        if extra_tokens:
            extra_token_dict = {'additional_special_tokens': extra_tokens}
            self.num_extra_tokens = self.encoder.add_special_tokens(extra_token_dict)
        else:
            self.num_extra_tokens = 0
        self.vocab_size = len(self.encoder.get_vocab())
        self.pad_token_id = self.encoder.pad_token_id
        self.__numcheck = ["eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", 
                           "zehn", "zwölf", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig",
                           "siebzig", "achtzig", "neunzig", "hundert", "tausend", "illion", "illiarde"]
        
    def preprocess(self, src_text):
        doc = self.nlp(src_text)
        new_tokens = []
        for token in doc:
            # If the token is part of a PERSON entity, mark as [NAME]
            if token.ent_type_ == "PER":
                new_tokens.append("[NAME]")
            # If the token looks like a number, or contains digits, or matches one of our number words:
            elif token.like_num or any(ch.isdigit() for ch in token.text) or any(sub in token.text.lower() for sub in self.__numcheck):
                new_tokens.append("[NUM]")
            else:
                new_tokens.append(token.text)
        # Join tokens back into a string.S
        sentence = " ".join(new_tokens)
        sentence = re.sub(r'\d.*', '', sentence)
        sentence = re.sub(r'[/^*|:!?.]', '', sentence)
        return sentence

    def encode(self, src_text, max_length=32):
        # Preprocess the text
        preprocessed_text = self.preprocess(src_text)
        # Encode the preprocessed text. We set add_special_tokens=False because we want to manually append [EOS].
        token_ids = self.encoder.encode(preprocessed_text, add_special_tokens=False, max_length=max_length, truncation=True)
        # Append the EOS token ID (make sure [EOS] is one of the added special tokens)
        eos_id = self.encoder.convert_tokens_to_ids("[EOS]")
        token_ids.append(eos_id)
        return token_ids


def clean_token(token):
    if type(token) is list:
        token = token[0]
    if "$NUM" in token:
        return "<num>"
    if "$NAME" in token:
        return "<name>"

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
