# import json
# import os
# import sys
# import torch
# import compress_fasttext
# import torch.nn.functional as F
# import speech_recognition as sr

# from dataset import DgsDataset
# from seq2seq import Seq2Seq, clean_token


# class Speech2DGS:
#     def __init__(self):
#         self.dir = os.path.dirname(os.path.abspath(sys.argv[0]))
#         self.translator = self.load_model()
#         self.embedder = self.load_embedder()
#         self.blender_vocabulary = self.load_blender_vocab()

#     def translate(self, sentence):
#         with torch.no_grad():
#             if "hallo" in sentence:
#                 pass
#             input_tensor = self.translator.sentence2vec(sentence)
#             outputs, _ = self.translator(input_tensor)
#             decoded_words = self.translator.decode_sequence(outputs)
#             blender_compatible = self.to_blender_tokens(decoded_words[:-1])
#         return blender_compatible

#     def to_blender_tokens(self, tokens):
#         for i, token in enumerate(tokens):
#             token = clean_token(token)
#             if token not in self.blender_vocabulary['words']:
#                 embeddings = torch.tensor(self.embedder[clean_token(token)]).unsqueeze(0)
#                 similarities = F.cosine_similarity(embeddings, self.blender_vocabulary['embeddings'], dim=-1)
#                 idx = int(similarities.argmax())
#                 tokens[i] = self.blender_vocabulary['words'][idx]
#             else:
#                 tokens[i] = token
#         return tokens
    
#     def load_model(self):
#         with open(os.path.join(self.dir, 'vocab/model_tokens.json'), 'r') as f:
#             vocabulary = json.load(f)
#         embedder_file = os.path.join(self.dir, 'models/word_embedding/cc.de.100.reduced.bin')
#         s2s = Seq2Seq(vocabulary, embedder_file, max_iter=20, hidden_size=128)
#         s2s.load_state_dict(torch.load(os.path.join(self.dir, "state_dict_v2.pt")))
#         s2s.eval()
#         return s2s

#     def load_embedder(self):
#         return compress_fasttext.models.CompressedFastTextKeyedVectors.load(os.path.join(self.dir, 'models/word_embedding/cc.de.100.reduced.bin'))

#     def load_blender_vocab(self):
#         with open(os.path.join(self.dir, 'blender/embedded_vocab.json'), 'r') as f:
#             d = json.load(f)
#         d['embeddings'] = torch.tensor(d['embeddings'])
#         return d

#     def write_to_file(self, sentence, tokens):
#         with open(os.path.join(self.dir, 'blender/NNoutput.txt'), 'w', encoding="utf-8") as f:
#             f.write(sentence + '\n')
#             for token in tokens:
#                 f.write(token + '\n')


# def live_speech2text(sdgs):
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)
#         print("Listening...")

#     while True:
#         with microphone as source:
#             audio = recognizer.listen(source)
#         try:
#             sentence = recognizer.recognize_google(audio, language="de-DE")
#             print(sentence)
#             if sentence:
#                 dgs = sdgs.translate(sentence)
#                 print("Translation: ", dgs)
#                 sdgs.write_to_file(sentence, dgs)
#         except sr.UnknownValueError:
#             print("Could not understand audio")
#         except sr.RequestError as e:
#             print("Could not request results; {0}".format(e))

# def main():
#     sdgs = Speech2DGS()
#     live_speech2text(sdgs)
#     # dgs = sdgs.translate(example_sentence)
#     # print(dgs)
#     # write_to_file(dgs)


# if __name__ == '__main__':
#     main()

import json
import os
import sys
import torch
import torch.nn.functional as F
import speech_recognition as sr
from transformers import AutoTokenizer

# Import the new Transformer model and cleaning function.
from transformer import Transformer, clean_token

# Define special tokens (must match those used during training)
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Greedy decoding function (adapted from the training notebook)
def decode(model, src_tensor, max_len, german_tokenizer, dgs_token2idx, dgs_idx2token, device):
    model.eval()
    # Encode the source: shape (src_seq_len, 1)
    src = src_tensor  # already tokenized and in shape (seq_len, 1)
    # Apply source embedding and positional encoding, then run encoder.
    memory = model.encoder(model.pos_encoder(model.src_embedding(src) * (model.model_dim ** 0.5)))
    tgt_indices = [dgs_token2idx[SOS_TOKEN]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(1)  # (tgt_seq_len, 1)
        # Embed and add positional encoding.
        tgt_embedded = model.tgt_embedding(tgt_tensor) * (model.model_dim ** 0.5)
        tgt_embedded = model.pos_decoder(tgt_embedded)
        # Create target mask for the current decoder input.
        tgt_seq_len = tgt_tensor.size(0)
        tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        # Run the decoder (we ignore memory masks here for simplicity).
        out = model.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
        out = model.fc_out(out)
        # Get the last token's prediction.
        prob = out[-1, 0]
        next_token = int(torch.argmax(prob).item())
        tgt_indices.append(next_token)
        if next_token == dgs_token2idx[EOS_TOKEN]:
            break
    # Convert indices back to tokens (exclude SOS and EOS)
    decoded_tokens = [dgs_idx2token[str(idx)] for idx in tgt_indices if idx not in {dgs_token2idx[SOS_TOKEN], dgs_token2idx[EOS_TOKEN]}]
    return decoded_tokens

class Speech2DGS:
    def __init__(self):
        self.dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        # Initialize the pretrained German tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
        
        # Load the target (DGS) vocabulary.
        # Expecting a JSON file with both token2idx and idx2token.
        with open(os.path.join(self.dir, 'dgs_vocab.json'), 'r', encoding="utf-8") as f:
            vocab = json.load(f)
        self.dgs_token2idx = vocab["token2idx"]
        self.dgs_idx2token = vocab["idx2token"]
        
        # Instantiate and load the new Transformer model.
        SRC_VOCAB_SIZE = self.tokenizer.vocab_size
        TGT_VOCAB_SIZE = len(self.dgs_token2idx)
        # Hyperparameters must match those used in training.
        MODEL_DIM = 256
        NUM_HEADS = 8
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3
        FF_DIM = 1024
        DROPOUT = 0.1
        
        self.translator = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                      model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                                      num_encoder_layers=NUM_ENCODER_LAYERS,
                                      num_decoder_layers=NUM_DECODER_LAYERS,
                                      ff_dim=FF_DIM, dropout=DROPOUT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator.to(device)
        try:
            self.translator.load_state_dict(torch.load(os.path.join(self.dir, "transformer_v1.pt"), map_location=device))
        except Exception as e:
            print("Could not load model state, training from scratch. Error:", e)
        self.translator.eval()
        self.device = device
        self.embedder = self.load_embedder()
        self.blender_vocabulary = self.load_blender_vocab()

    def translate(self, sentence):
        with torch.no_grad():
            # Tokenize the German sentence.
            src_tokens = self.tokenizer.encode(sentence, add_special_tokens=True)
            src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
            # Use greedy decoding to generate DGS tokens.
            decoded_tokens = decode(self.translator, src_tensor, max_len=20,
                                           german_tokenizer=self.tokenizer,
                                           dgs_token2idx=self.dgs_token2idx,
                                           dgs_idx2token=self.dgs_idx2token,
                                           device=self.device)
            blender_compatible = self.to_blender_tokens(decoded_tokens)
        return blender_compatible

    def to_blender_tokens(self, tokens):
        for i, token in enumerate(tokens):
            token = clean_token(token)
            if token not in self.blender_vocabulary['words']:
                embeddings = torch.tensor(self.embedder[clean_token(token)]).unsqueeze(0)
                similarities = F.cosine_similarity(embeddings, self.blender_vocabulary['embeddings'], dim=-1)
                idx = int(similarities.argmax())
                tokens[i] = self.blender_vocabulary['words'][idx]
            else:
                tokens[i] = token
        return tokens

    def load_embedder(self):
        import compress_fasttext
        embedder_file = os.path.join(self.dir, 'models/word_embedding/cc.de.100.reduced.bin')
        return compress_fasttext.models.CompressedFastTextKeyedVectors.load(embedder_file)

    def load_blender_vocab(self):
        with open(os.path.join(self.dir, 'blender/embedded_vocab.json'), 'r', encoding="utf-8") as f:
            d = json.load(f)
        d['embeddings'] = torch.tensor(d['embeddings'])
        return d

    def write_to_file(self, sentence, tokens):
        with open(os.path.join(self.dir, 'blender/NNoutput.txt'), 'w', encoding="utf-8") as f:
            f.write(sentence + '\n')
            for token in tokens:
                f.write(token + '\n')

def live_speech2text(sdgs):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")

    while True:
        with microphone as source:
            audio = recognizer.listen(source)
        try:
            sentence = recognizer.recognize_google(audio, language="de-DE")
            print(sentence)
            if sentence:
                dgs = sdgs.translate(sentence)
                print("Translation: ", dgs)
                sdgs.write_to_file(sentence, dgs)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

def main():
    sdgs = Speech2DGS()
    live_speech2text(sdgs)

if __name__ == '__main__':
    main()

