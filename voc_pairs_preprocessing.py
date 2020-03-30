from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import random
import re
import os
import unicodedata
from io import open
import itertools

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}  # mapping from words to indexes
        self.word2count = {}  # count of each word
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}  # reverse mapping of indexes to words
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        """Add all words of a sentence to the vocabulary"""
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """Add one word to the vocabulary"""
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        """Remove words below a certain count threshold"""
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count default tokens

        for word in keep_words:
            self.addWord(word)

    def indexesFromSentence(self, sentence):
        """Convert words of a sentence to their indexes"""
        return [self.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    """Zero-pad a matrix of word indexes"""
    # This iterator falls under the category of Terminating Iterators.
    # It prints the values of iterables alternatively in sequence.
    # If one of the iterables is printed fully,
    # the remaining values are filled by the values assigned to fillvalue parameter.
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def unicodeToAscii(s):
    """Turn a Unicode string to plain ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """Lowercase, trim, and remove non-letter characters"""
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def readVocs(datafile):
    """Read query/response pairs, and normalize them"""
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    return pairs


def filterPair(p, max_length):
    """Returns True if both sentences in a pair 'p' are under or equal the max_length threshold"""
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) <= max_length and len(p[1].split(' ')) <= max_length


def filterPairs(pairs, max_length):
    """Filter pairs using filterPair condition"""
    return [pair for pair in pairs if filterPair(pair, max_length)]


def trimRareWords(voc, pairs, min_count):
    """Trim words used under the min_count from the voc"""
    voc.trim(min_count)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


def loadPrepareData(datafile, max_length=10, min_count=3):
    """Load, normalize and filter the pairs of lines in "datafile" and
    create a Voc object populated with the words in the lines.
    Args:
    - datafile: path to the file to read containing pairs of sentences, one pair each line.
                Each line is separated by the \t delimiter
    - max_length: maximum sentence length to consider
    Return:
    - voc: Voc object populated with the words in the lines
    - pairs: pairs of lines in "datafile", normalized and filtered
    """
    print("Start preparing training data ...")
    pairs = readVocs(datafile)
    voc = Voc()
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, max_length)
    print("Filtered to {!s} sentence pairs".format(len(pairs)))
    print("Creating vocabulary...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    pairs = trimRareWords(voc, pairs, min_count)
    print("Counted words:", voc.num_words)
    return voc, pairs


def binaryMatrix(l, value=PAD_token):
    """Create a binary mask of the input sequence"""
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(l, voc):
    """Returns padded input sequence tensor and lengths"""
    indexes_batch = [voc.indexesFromSentence(sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(l, voc):
    """Returns padded target sequence tensor, padding mask, and max target length"""
    indexes_batch = [voc.indexesFromSentence(sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    """Returns all items for a given batch of pairs
    Args:
    - voc: Voc object
    - pair_batch: batch of pairs
    Return:
    - inp: padded and transposed input batch ordered by length
    - lengths: lengths of the sentences in the input batch
    - output: padded and transposed output batch
    - mask: binary mask of the output
    - max_target_len: length of the longest sequence in the output batch"""
    # order pairs according to the length of the first sentence in each pair
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    # pad and transpose the input
    inp, lengths = inputVar(input_batch, voc)
    # pad and transpose the input and build its corresponding binary mask
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


if __name__ == '__main__':

    MAX_LENGTH = 20  # Maximum sentence length to consider
    MIN_COUNT = 0
    corpus_location = "data/cornell movie-dialogs corpus"
    filename = "formatted_movie_lines.txt"

    filename_path = os.path.join(corpus_location, filename)

    voc, pairs = loadPrepareData(filename_path, MAX_LENGTH, MIN_COUNT)
    # Print some pairs to validate
    print("\nPairs:", len(pairs))
    for pair in pairs[:10]:
        print(pair)

    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)
