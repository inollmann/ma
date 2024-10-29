import re
import pickle
import itertools
# import compress_fasttext
from collections import Counter, defaultdict
from torch.utils.data import Dataset
from seq2seq import Seq2Seq, clean_token


def read_data(file):
    with open(file, encoding='utf-8') as f:
        data = f.read().splitlines()
    pairs = [re.split(r'\s{2,}', line) for line in data]

    return pairs


def read_pkl(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    to_remove = []

    for idx, (de, dgs) in enumerate(zip(data['de'], data['dgs'])):
        if len(de) <= 1 or not dgs or de.isspace(): # or contains_multiple_sentences(de)
            to_remove.append(idx)

    for idx in sorted(to_remove, reverse=True):
        del data['transcript'][idx]
        del data['index'][idx]
        del data['de'][idx]
        del data['dgs'][idx]
        del data['mouth'][idx]

    return data['transcript'], data['index'], data['de'], data['dgs'], data['mouth']


def contains_multiple_sentences(text):
    sentence_endings = re.findall(r'[.!?](?:\s|$)', text)
    return len(sentence_endings) > 1


def get_tokens(sentences):
    # sentences = [entry.split() for entry in sentences]
    tokens = list(itertools.chain.from_iterable(sentences))
    counter = Counter(tokens)
    return ["<SOS>", "<EOS>"] + sorted(counter, key=counter.get, reverse=True)


def get_max_tokens(sentences):
    max_tokens = 0
    for tokens in sentences:
        # tokens = sentence.split()
        if len(tokens) > max_tokens:
            max_tokens = len(tokens)
    return max_tokens


class DgsDataset(Dataset):

    def __init__(self, dataset_file, embedder_file, simplify=True):
        self.simplify = simplify
        self.transcript, self.index, self.de, self.dgs, self.mouth = read_pkl(dataset_file)
        self.vocabulary = get_tokens(self.dgs)
        self.vocab_groups = self.simplify_vocabulary()
        if self.simplify:
            self.vocabulary = self.get_from_groups()
        self.max_tokens = get_max_tokens(self.dgs)
        self.model = Seq2Seq(self.vocabulary, embedder_file, self.max_tokens)
        self.transform_de = self.model.sentence2vec
        self.transform_dgs = self.model.encode_dgs


    def __len__(self):
        return len(self.index)


    def __getitem__(self, idx):
        de_string = self.de[idx]
        dgs_list = self.dgs[idx]
        if self.simplify:
            dgs_list = self.simplify_dgs(dgs_list)
        de_tensor = self.transform_de(de_string)
        dgs_tensor = self.transform_dgs(dgs_list)

        return de_tensor, dgs_tensor, de_string, dgs_list, idx


    def simplify_vocabulary(self):
        groups = defaultdict(list)
        for token in self.vocabulary:
            cleaned = clean_token(token)
            groups[cleaned].append(token)

        return groups


    def token2most_common(self, token):
        cleaned = clean_token(token)
        if cleaned in self.vocab_groups.keys():
            token = self.vocab_groups[cleaned][0]

        return token
    
    
    def simplify_dgs(self, sentence):
        # simplified = ""
        # for token in sentence.split():
            # simplified = simplified + self.token2most_common(token) + " "
        for i, token in enumerate(sentence):
            sentence[i] = self.token2most_common(token)
        
        return sentence #simplified


    def get_from_groups(self):
        reduced_vocab = []
        for _, tokens in self.vocab_groups.items():
            reduced_vocab.append(tokens[0])

        return reduced_vocab
