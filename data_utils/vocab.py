import torch
from data_utils.vector import Vectors
from data_utils.vector import pretrained_aliases
from data_utils.utils import preprocess_sentence
from collections import defaultdict, Counter
import logging
import six
import json
from typing import List

logger = logging.getLogger(__name__)

def _default_unk_index():
    return 0

class Vocab:
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, json_dirs, min_freq=1, vectors=None, unk_init=None, vectors_cache=None):
        self.make_vocab(json_dirs)
        counter = self.freqs.copy()
        min_freq = max(min_freq, 1)

        self.padding_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.special_tokens = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        itos = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq:
                break
            itos.append(word)
        self.itos = {i: w for i, w in enumerate(itos)}

        self.stoi = defaultdict(_default_unk_index)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in self.itos.items()})

        self.padding_idx = self.stoi[self.padding_token]
        self.bos_idx = self.stoi[self.bos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]
        self.special_ids = [self.padding_idx, self.bos_idx, self.eos_idx, self.unk_idx]

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def make_vocab(self, json_dirs: List[str]):
        self.freqs = Counter()
        self.tags = set()
        self.max_sentence_length = 0
        
        for json_dir in json_dirs:
            data = json.load(open(json_dir, "r"))
            for sample in data:
                sentence = preprocess_sentence(sample["sentence"])
                self.freqs.update(sentence)
                self.tags.update(sample["tag"])
                if len(sentence) + 2 > self.max_sentence_length:
                    self.max_sentence_length = len(sentence) + 2

        self.tags = list(self.tags)

    def encode_sentence(self, sentence):
        """ Turn a sentence into a vector of indices """
        vec = torch.ones(self.max_sentence_length).long() * self.padding_idx
        for idx, token in enumerate([self.bos_token] + sentence + [self.eos_token]):
            vec[idx] = self.stoi[token]

        return vec

    def encode_tag(self, tags):
        """ Turn a tag of sentence into vector of indices """
        tag_indices = torch.ones(self.max_sentence_length).long() * self.tags.index("O")
        for ith, tag in enumerate(["O"] + tags + ["O"]):
            tag_indices[ith] = self.tags.index(tag)
        
        return tag_indices

    def decode_sentence(self, sentence_vecs):
        sentences = []
        for vec in sentence_vecs:
            sentences.append([self.itos[idx] for idx in vec.tolist() if idx not in self.special_ids])

        return sentences

    def decode_tag(self, tag_vecs, lens):
        tags = []
        for vec, len in zip(tag_vecs.tolist(), lens):
            tags.append([self.tags[idx] for idx in vec[1:len]])

        return tags

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def load_vectors(self, vectors, **kwargs):
        """
        Arguments:
            vectors: one of or a list containing instantiations of the
                fastText, PhoW2V or Vectors classes. Alternatively, one
                of or a list of available pretrained vectors:
                fasttext.vi.300d
                phow2v.syllable.100d
                phow2v.syllable.300d
                phow2v.word.100d
                phow2v.word.300d
            Remaining keyword arguments: Passed to the constructor of Vectors classes.
        """
        if not isinstance(vectors, list):
            vectors = [vectors]
        for idx, vector in enumerate(vectors):
            if six.PY2 and isinstance(vector, str):
                vector = six.text_type(vector)
            if isinstance(vector, six.string_types):
                # Convert the string pretrained vector identifier
                # to a Vectors object
                if vector not in pretrained_aliases:
                    raise ValueError(
                        "Got string input vector {}, but allowed pretrained "
                        "vectors are {}".format(
                            vector, list(pretrained_aliases.keys())))
                vectors[idx] = pretrained_aliases[vector](**kwargs)
            elif not isinstance(vector, Vectors):
                raise ValueError(
                    "Got input vectors of type {}, expected str or "
                    "Vectors object".format(type(vector)))

        tot_dim = sum(v.dim for v in vectors)
        self.vectors = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, vectors, dim, unk_init=torch.Tensor.zero_):
        """
        Set the vectors for the Vocab instance from a collection of Tensors.
        Arguments:
            stoi: A dictionary of string to the index of the associated vector
                in the `vectors` input argument.
            vectors: An indexed iterable (or other structure supporting __getitem__) that
                given an input index, returns a FloatTensor representing the vector
                for the token associated with the index. For example,
                vector[stoi["string"]] should return the vector for "string".
            dim: The dimensionality of the vectors.
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
        """
        self.vectors = torch.Tensor(len(self), dim)
        for i, token in enumerate(self.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                self.vectors[i] = vectors[wv_index]
            else:
                self.vectors[i] = unk_init(self.vectors[i])