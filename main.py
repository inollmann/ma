import json
import os
import sys
import torch
import compress_fasttext
import torch.nn.functional as F
import speech_recognition as sr

from dataset import DgsDataset
from seq2seq import Seq2Seq, clean_token


class Speech2DGS:
    def __init__(self):
        self.dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        self.translator = self.load_model()
        self.embedder = self.load_embedder()
        self.blender_vocabulary = self.load_blender_vocab()

    def translate(self, sentence):
        with torch.no_grad():
            if "hallo" in sentence:
                pass
            input_tensor = self.translator.sentence2vec(sentence)
            outputs, _ = self.translator(input_tensor)
            decoded_words = self.translator.decode_sequence(outputs)
            blender_compatible = self.to_blender_tokens(decoded_words[:-1])
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
    
    def load_model(self):
        with open(os.path.join(self.dir, 'vocab/model_tokens.json'), 'r') as f:
            vocabulary = json.load(f)
        embedder_file = os.path.join(self.dir, 'models/word_embedding/cc.de.100.reduced.bin')
        s2s = Seq2Seq(vocabulary, embedder_file, max_iter=20, hidden_size=128)
        s2s.load_state_dict(torch.load(os.path.join(self.dir, "state_dict_v2.pt")))
        s2s.eval()
        return s2s

    def load_embedder(self):
        return compress_fasttext.models.CompressedFastTextKeyedVectors.load(os.path.join(self.dir, 'models/word_embedding/cc.de.100.reduced.bin'))

    def load_blender_vocab(self):
        with open(os.path.join(self.dir, 'blender/embedded_vocab.json'), 'r') as f:
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
    # dgs = sdgs.translate(example_sentence)
    # print(dgs)
    # write_to_file(dgs)


if __name__ == '__main__':
    main()
