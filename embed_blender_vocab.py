import compress_fasttext
import json
import os

def main():
    ft = compress_fasttext.models.CompressedFastTextKeyedVectors.load('./models/word_embedding/cc.de.100.reduced.bin')

    embeddings_list = []
    word_list = []
    for file in os.listdir('./blender/vocab'):
        word = file.split('.')[0]
        word_list.append(word)
        embeddings = ft[word]
        embeddings_list.append(embeddings.tolist())

    word_embeddings = {'words': word_list, 'embeddings': embeddings_list}

    with open('./blender/embedded_vocab.json', 'w') as outfile:
        json.dump(word_embeddings, outfile)


if __name__ == "__main__":
    main()
