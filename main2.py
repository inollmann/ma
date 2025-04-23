import json
import os
import sys
import torch
import pyaudio
import argparse
import torch.nn.functional as F
import speech_recognition as sr
# from transformers import AutoTokenizer

from jupy.transformer import Transformer, Tokenizer, clean_token

# Special tokens
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
NAME_TOKEN = "<name>"
NUM_TOKEN = "<num>"
UNK_TOKEN = "<unk>"

articles = ["ein", "eine", "eines", "einem", "einer", "einen", "der", "die", "das", "dem", "des", "dem", "den"]

def decode(model, src_tensor, max_len, dgs_token2idx, dgs_idx2token, device, remove_rep=True):
    model.eval()
    # Encode source (src_seq_len, 1)
    src = src_tensor
    # Source embedding, positional encoding, encoder
    memory = model.encoder(model.pos_encoder(model.src_embedding(src) * (model.model_dim ** 0.5)))
    tgt_indices = [dgs_token2idx[SOS_TOKEN]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(1)  # (tgt_seq_len, 1)
        # Embedding and positional encoding
        tgt_embedded = model.tgt_embedding(tgt_tensor) * (model.model_dim ** 0.5)
        tgt_embedded = model.pos_decoder(tgt_embedded)
        # Target mask for current decoder input
        tgt_seq_len = tgt_tensor.size(0)
        tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        # Decoder
        out = model.decoder(tgt_embedded, memory, tgt_mask=tgt_mask)
        out = model.fc_out(out)
        # Get token prediction
        prob = out[-1, 0]
        next_token = int(torch.argmax(prob).item())
        tgt_indices.append(next_token)
        if next_token == dgs_token2idx[EOS_TOKEN]:
            break

    if remove_rep:
        tgt_indices = remove_repetitions(tgt_indices)
        tgt_indices = remove_alternating_repetitions(tgt_indices)

    # Convert indices back to tokens (exclude SOS and EOS)
    decoded_tokens = [dgs_idx2token[str(idx)] for idx in tgt_indices if idx not in {dgs_token2idx[SOS_TOKEN], dgs_token2idx[EOS_TOKEN]}]
    return decoded_tokens

def remove_repetitions(tokens):
    if not tokens:
        return []
    cleaned = [tokens[0]]
    for token in tokens[1:]:
        if token != cleaned[-1]:
            cleaned.append(token)
    return cleaned

def remove_alternating_repetitions(tokens):
    n = len(tokens)
    if n < 4:
        return tokens

    for start in range(n - 3):
        A = tokens[start]
        B = tokens[start + 1]
        if A == B:
            continue
        valid = True
        for i in range(start, n):
            expected = A if (i - start) % 2 == 0 else B
            if tokens[i] != expected:
                valid = False
                break
        if valid:
            return tokens[:start + 2]
    return tokens

class Speech2DGS:
    def __init__(self, speech2text, model_config=[61, 1, 1, 64, 1000], mic_idx=1):

        self.dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        # Initialize tokenizer
        self.tokenizer = Tokenizer()

        # (Offline) Recognizer configuration
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=mic_idx)
        match speech2text:
            case "whisper":
                self.recognize = lambda audio: self.recognizer.recognize_whisper(audio, language="german", model="medium")
            case "google":
                self.recognize = lambda audio: self.recognizer.recognize_google(audio, language="de-DE")
            case "openai_api":
                with open("key.txt", "r") as file:
                    os.environ["OPENAI_API_KEY"] = file.read().strip()
                self.recognize = lambda audio: self.recognizer.recognize_whisper_api(audio)
            case _:
                raise ValueError(f"Unknown recognizer option: {speech2text}")
        
        # Load target (DGS) vocabulary (JSON with token2idx and idx2token)
        with open(os.path.join(self.dir, 'dgs_vocab.json'), 'r', encoding="utf-8") as f:
            vocab = json.load(f)
        self.dgs_token2idx = vocab["token2idx"]
        self.dgs_idx2token = vocab["idx2token"]
        
        # Load new transformer model
        SRC_VOCAB_SIZE = self.tokenizer.vocab_size
        TGT_VOCAB_SIZE = len(self.dgs_token2idx)
        # Hyperparameters from training
        MODEL_DIM = model_config[0]
        NUM_HEADS = 8
        NUM_ENCODER_LAYERS = model_config[1]
        NUM_DECODER_LAYERS = model_config[2]
        FF_DIM = model_config[3]
        DROPOUT = 0
        
        self.translator = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                      model_dim=MODEL_DIM, num_heads=NUM_HEADS,
                                      num_encoder_layers=NUM_ENCODER_LAYERS,
                                      num_decoder_layers=NUM_DECODER_LAYERS,
                                      ff_dim=FF_DIM, dropout=DROPOUT)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator.to(device)
        state_dict = f"d{model_config[0]}_el{model_config[1]}_dl{model_config[2]}_ff{model_config[3]}-ep{model_config[4]}.pt"
        try:
            self.translator.load_state_dict(torch.load(os.path.join(self.dir, state_dict), map_location=device))
        except Exception as e:
            print("Could not load model state; loading random states. Error:", e)
        self.translator.eval()
        self.device = device
        self.embedder = self.load_embedder()
        self.blender_vocabulary = self.load_blender_vocab()

    def translate(self, sentence):
        with torch.no_grad():
            # Tokenize German sentence
            src_tokens = self.tokenizer.encode(sentence)
            src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
            # Run model decoding
            #print(src_tokens, len(sentence.split()))
            #print(self.tokenizer.encoder.convert_ids_to_tokens(src_tokens))
            check = sentence.lower().split()
            if len(check) == 1:
                decoded_tokens = check
            elif len(check) == 2 and check[0] in articles:
                decoded_tokens = [check[1]]
            else:
                decoded_tokens = decode(self.translator, src_tensor, max_len=20,
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

    def live_speech2text(self, sdgs):
        # recognizer = sr.Recognizer()
        # microphone = sr.Microphone()

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening...")

        while True:
            with self.microphone as source:
                audio = self.recognizer.listen(source)
            try:
                sentence = self.recognize(audio)
                print(sentence)
                if sentence:
                    dgs = sdgs.translate(sentence)
                    print("Translation: ", dgs)
                    sdgs.write_to_file(sentence, dgs)
            except sr.UnknownValueError:
                print("Could not understand audio", end='\x1b[1K\r')
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))


def main():
    # Arguments
    parser = argparse.ArgumentParser(description="Use custom microphone")
    parser.add_argument("--mic_idx", type=int, default=None, help="Index of the microphone device")
    args = parser.parse_args()

    # Config mic
    if args.mic_idx is not None:
        device_index = int(args.mic_idx)
    else:
        device_index=2
    audio = pyaudio.PyAudio()
    device_info = audio.get_device_info_by_index(device_index)
    device_name = device_info["name"]
    print(f"Using microphone {device_index}: {device_name})")

    sdgs = Speech2DGS(speech2text="google", model_config=[64, 1, 1, 64, 1000], mic_idx=device_index)     # options: "google", "whisper", "openai_api" (requires API key)
    sdgs.live_speech2text(sdgs)

if __name__ == '__main__':
    print("Initiating program...")
    main()