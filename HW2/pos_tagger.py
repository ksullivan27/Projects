import copy
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import time
import string
from tqdm import tqdm
from word2number import w2n
from utils import *

""" Contains the part of speech tagger class. """


def get_test_pred(data, model):
    processes = 10
    n = len(data)
    k = n // processes

    start = time.time()
    pool = Pool(processes=processes)
    res = []

    for i in range(0, n, k):
        res.append(
            pool.apply_async(infer_sentences,
                             [model, data[i:i + k], i]))

    temp_lst = [r.get(timeout=None) for r in res]
    temp_dict = {key: item for temp_pair in temp_lst for key, item in temp_pair.items()}
    temp = [temp_dict[i] for i in range(len(temp_dict))]
    test_predictions = [i for sublist in temp for i in sublist]

    print(f"Inference Runtime: {(time.time() - start) / 60} minutes.")

    write_csv(test_predictions, "test_y.csv")

    print('Predictions Complete')


def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is.
    
    As per the write-up, you may find it faster to use multiprocessing (code included).
    
    """
    processes = 10
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n // processes
    n_tokens = sum([len(d) for d in sentences])
    token_set = set(model.all_tokens)
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in token_set])
    predictions = {i: None for i in range(n)}
    probabilities = {i: None for i in range(n)}

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(
            pool.apply_async(infer_sentences,
                             [model, sentences[i:i + k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time() - start) / 60} minutes.")

    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob,
                                    [model, sentences[i:i + k], tags[i:i + k],
                                     i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(
        f"Probability Estimation Runtime: {(time.time() - start) / 60} minutes.")

    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if
                     tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum(
        [1 for i in range(n) for j in range(len(sentences[i])) if
         tags[i][j] == predictions[i][j] and sentences[i][
             j] not in model.word2idx.keys()]) / unk_n_tokens
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx + 1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc / num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values()) / n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))

    return whole_sent_acc / num_whole_sent, token_acc, sum(
        probabilities.values()) / n


def suffix_groups(token, prev):
    """
    Classify unknown words according to suffixes.
    """

    # Check if proper noun (singular vs. plural)
    if prev != '<STOP_WORD>' and token[0].isupper():
        if token[-1].lower() == 's':
            # print(token, "NNPS")
            return "NNPS"
        else:
            # print(token, "NNP")
            return "NNP"

    # Check adverb
    elif token[-2:] == "ly":
        return "RB"

    # Check adjective
    elif token[-4:] == "able":
        return "JJ"

    # Check if it's a cardinal number
    elif type(token) == int or type(token) == float:
        return "CD"

    # Check if the word is a spelled out number
    try:
        w2n.word_to_num(token)
        return "CD"
    except:
        pass

    try:
        w2n.word_to_num(token[0])
        return "CD"
    except:
        pass

    try:
        w2n.word_to_num(token[-1])
        return "CD"
    except:
        pass

    return None


def linear_interpolation(trigrams_f, bigrams_f, unigrams_f, n_tags):
    """
    Strategy for finding the best lambdas as detailed in this paper: https://aclanthology.org/A00-1031.pdf
    """
    lambdas = [0, 0, 0]
    N = sum(unigrams_f)
    for index, trigram in enumerate(trigrams_f):
        if (bigrams_f[index % (n_tags ** 2)] - 1) == 0:
            trigram_bigram = 0
        else:
            trigram_bigram = float(trigram - 1) / (bigrams_f[index % (n_tags ** 2)] - 1)
        if (unigrams_f[index % n_tags] - 1) == 0:
            bigram_unigram = 0
        else:
            bigram_unigram = float(bigrams_f[index % (n_tags ** 2)] - 1) / (unigrams_f[index % n_tags] - 1)
        if (N - 1) == 0:
            unigram_n = 0
        else:
            unigram_n = float(unigrams_f[index % n_tags] - 1) / (N - 1)
        max_value = np.argmax([trigram_bigram,
                               bigram_unigram,
                               unigram_n])
        if max_value == 0:
            lambdas[0] += trigram
        elif max_value == 1:
            lambdas[1] += trigram
        else:
            lambdas[2] += trigram
    return tuple([float(i) / sum(lambdas) for i in lambdas])


class POSTagger:
    """
    Main class for the POS tagger.
    """

    def __init__(self,
                 data,
                 unknown_token_handling: bool = False,
                 smoothing_strategy: str = None,
                 model: str = 'unigram',
                 method: str = 'greedy',
                 suffix_bool: bool = False,
                 smooth_k: float = 0,
                 unk_threshold: int = 3,
                 beam_k: int = 2):
        """Initializes the tagger model parameters and anything else necessary."""
        self.data = data

        self.all_tokens = list(set([t for token in data[0] for t in token]))
        self.all_tokens.append('<STOP_WORD>')
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.all_tags.append('<STOP_WORD>')

        self.n_tokens = len(self.all_tokens)
        self.n_tags = len(self.all_tags)
        self.n_docs = len(self.data[0])

        self.tokens = [t for token in data[0] for t in token]
        self.tags = [t for tag in data[1] for t in tag]

        self.token_counts = {t: 0 for t in self.all_tokens}
        for token in self.tokens:
            self.token_counts[token] += 1

        self.tag_counts = {t: 0 for t in self.all_tags}
        for tag in self.tags:
            self.tag_counts[tag] += 1

        self.word2idx = {self.all_tokens[i]: i for i in
                         range(len(self.all_tokens))}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        self.tag2idx = {self.all_tags[i]: i for i in range(len(self.all_tags))}
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        self.grams = None
        self.emissions = None
        self.e_matrix = None
        self.t_matrix = None

        self.predictions = {i: None for i in range(self.n_docs)}
        self.probabilities = {i: None for i in range(self.n_docs)}

        self.unknown_token_handling = unknown_token_handling

        self.smoothing_strategy = smoothing_strategy

        self.model = model

        self.method = method

        self.suffix_bool = suffix_bool

        self.smooth_k = smooth_k

        self.unk_threshold = unk_threshold

        self.beam_k = beam_k

    def unknown_tokens(self, threshold):
        """
        If a token occurs less than threshold times, replace it with unk.
        """
        # enter path to saved file (if it exists), otherwise run the below code
        my_file = Path("content/tokens_w_unk_4.csv")
        if not my_file.is_file():
            print('Swapping <unk> for tokens with fewer than "threshold" occurrences.')

            all_tokens_copy = self.all_tokens.copy()
            tokens_to_replace = set()
            for item in self.token_counts.items():
                if item[1] <= threshold:
                    tokens_to_replace.add(item[0])
                    all_tokens_copy.remove(item[0])

            processes = 10
            sentences = self.data[0]
            tags = self.data[1]
            n = len(sentences)
            k = n // processes

            pool = Pool(processes=processes)
            sent_adj = []
            for i in tqdm(range(0, n, k)):
                output = pool.apply_async(unk_replace,
                                          [sentences[i:i + k],
                                           tokens_to_replace]).get(timeout=None)
                sent_adj.append(output)

            sents_doc = [s for s in sent_adj]

            tokens_temp = [t for sent in sents_doc for t in sent]
            tokens = [t for sublist in tokens_temp for t in sublist]

            write_csv(tokens, "tokens_w_unk_4.csv")

        else:
            temp = pd.read_csv('tokens_w_unk_4.csv')
            temp = temp['tag'].tolist()

            tokens = ['-docstart-']
            for i in temp[1:]:
                if i == '-docstart-':
                    tokens.append('<STOP>')
                tokens.append(i)
            tokens.append('<STOP>')

        # combine the original token list along with the now <unk> replaced
        # list to get a list twice as large (this allows us to train on these
        # newly swapped <unk> values in addition to the non-swapped values
        self.tokens.extend(tokens)
        # double the tags as well
        self.tags.extend(self.tags)

        # update the tokens and tags set
        self.all_tokens = list(set(self.tokens))
        self.all_tokens.append('<STOP_WORD>')

        # update the unique tokens and tags counts
        self.n_tokens = len(self.all_tokens)
        self.n_tags = len(self.all_tags)

        # update the tokens counts dict
        self.token_counts = {t: 0 for t in self.all_tokens}
        for token in self.tokens:
            self.token_counts[token] += 1

        # update the tags counts dict
        self.tag_counts = {t: 0 for t in self.all_tags}
        for tag in self.tags:
            self.tag_counts[tag] += 1

        # update the tokens idx dicts
        self.word2idx = \
            {self.all_tokens[i]: i for i in range(len(self.all_tokens))}
        self.idx2word = \
            {v: k for k, v in self.word2idx.items()}

        # update the tags idx dicts
        self.tag2idx = \
            {self.all_tags[i]: i for i in range(len(self.all_tags))}
        self.idx2tag = \
            {v: k for k, v in self.tag2idx.items()}

    def add_k_smoothing(self, k, array, count):
        """
        Assumes every seen or unseen event occurs k times more than it did in training data.
        """
        if self.model == 'bigram':
            return (array + k) / (count + k * self.n_tags)
        elif self.model == 'trigram':
            return (array + k) / (count + k * self.n_tags ** 2)

    def get_unigrams(self):
        """
        Computes unigrams.
        Tip. Map each tag to an integer and store the unigrams in a numpy array.
        """
        output_array = np.array([0 for i in range(self.n_tags)])
        for tag in self.tags:
            output_array[self.tag2idx[tag]] += 1

        output_array = output_array.astype('float64')
        output_array /= len(output_array)

        return output_array

    def get_bigrams(self):
        """
        Computes bigrams. Map each tag to an integer and store the bigrams in a numpy array
        such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1).
        """
        tags_cleaned = ["<STOP_WORD>"]
        for index, token in enumerate(self.tokens):
            tags_cleaned.append(self.tags[index])
            if token in '.!?;' or token == '-docstart-':
                tags_cleaned.append("<STOP_WORD>")

        if tags_cleaned[-2] != "<STOP_WORD>":
            end_symb = tags_cleaned[-1]
            tags_cleaned[-1] = "."
            tags_cleaned.append("<STOP_WORD>")
            tags_cleaned.append(end_symb)

        output_array = np.array([0.0 for _ in range(self.n_tags ** 2)])

        prev_index = self.tag2idx[tags_cleaned[0]]
        for tag in tags_cleaned[1:]:
            current_index = self.tag2idx[tag]
            output_array[(prev_index * self.n_tags) + current_index] += 1
            prev_index = current_index

        for tag in list(self.tag2idx.keys()):
            count = tags_cleaned.count(tag)
            index = self.tag2idx[tag]
            if self.smoothing_strategy == 'add-k':
                k = self.smooth_k
                output_array[index * self.n_tags:index * self.n_tags + self.n_tags] = \
                    self.add_k_smoothing(k, output_array[index * self.n_tags:index * self.n_tags + self.n_tags], count)
            else:
                output_array[index * self.n_tags:index * self.n_tags + self.n_tags] /= count

        return output_array

    def get_trigrams(self):
        """
        Computes trigrams. Store in numpy array.
        """
        tags_cleaned = ["<STOP_WORD>", "<STOP_WORD>"]
        for index, token in enumerate(self.tokens):
            tags_cleaned.append(self.tags[index])
            if token in '.!?;' or token == '-docstart-':
                tags_cleaned.append("<STOP_WORD>")
                tags_cleaned.append("<STOP_WORD>")

        if tags_cleaned[-2] != "<STOP_WORD>":
            end_symb = tags_cleaned[-1]
            tags_cleaned[-1] = "."
            tags_cleaned.append("<STOP_WORD>")
            tags_cleaned.append("<STOP_WORD>")
            tags_cleaned.append(end_symb)

        output_array = np.array([0.0 for _ in range(self.n_tags ** 3)])

        prev_two_index = self.tag2idx[tags_cleaned[0]]
        prev_one_index = self.tag2idx[tags_cleaned[1]]
        for tag in tags_cleaned[2:]:
            current_index = self.tag2idx[tag]
            output_array[
                (prev_two_index * self.n_tags ** 2) +
                prev_one_index * self.n_tags + current_index] += 1
            prev_one_index, prev_two_index = current_index, prev_one_index

        output_array_temp = np.array([0.0 for _ in range(self.n_tags ** 2)])
        prev_index = self.tag2idx[tags_cleaned[0]]
        for tag in tags_cleaned[1:]:
            current_index = self.tag2idx[tag]
            output_array_temp[
                (prev_index * self.n_tags) + current_index] += 1
            prev_index = current_index

        for i in range(len(output_array)):
            if self.smoothing_strategy == 'add-k':
                k = self.smooth_k
                output_array[i] = \
                    self.add_k_smoothing(k, output_array[i], output_array_temp[i // self.n_tags])
            else:
                if output_array_temp[i // self.n_tags] == 0:
                    print('ZERO!')
                    output_array[i] = 0
                else:
                    output_array[i] /= output_array_temp[i // self.n_tags]

        return output_array

    def get_emissions(self):
        """
        Computes emission probabilities. Map each tag to an integer and each word in the vocabulary to an integer.
        Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag)
        """
        output_array = np.array(
            [0 for _ in range(self.n_tags * self.n_tokens)])

        for index, token in enumerate(self.tokens):
            tag = self.tags[index]
            tag_index = self.tag2idx[tag]
            token_index = self.word2idx[token]
            output_array[(tag_index * self.n_tokens) + token_index] += 1

        token_index = self.word2idx['<STOP_WORD>']
        tag_index = self.tag2idx['<STOP_WORD>']
        output_array[(tag_index * self.n_tokens) + token_index] = 1

        output_array = output_array.astype('float64')

        for i in range(0, self.n_tags * self.n_tokens, self.n_tokens):
            count = sum(output_array[i: i + self.n_tokens])
            if count == 0:
                count = 1
            output_array[i: i + self.n_tokens] /= count

        return output_array

    def train(self):
        """Trains the model by computing transition and emission probabilities."""
        if self.unknown_token_handling:
            self.unknown_tokens(self.unk_threshold)

        if self.model == 'unigram':
            self.grams = self.get_unigrams()
        elif self.model == 'bigram':
            self.grams = self.get_bigrams()
        elif self.model == 'trigram':
            self.grams = self.get_trigrams()

        self.emissions = self.get_emissions()
        self.e_matrix = self.emission_matrix()
        if self.model == 'bigram':
            self.t_matrix = self.transition_bi_matrix()
        elif self.model == 'trigram':
            self.t_matrix = self.transition_tri_matrix()

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition probabilities."""
        probability = 1
        if self.model == 'unigram':
            for index, tag in enumerate(tags):
                tag_index = self.tag2idx[tag]
                probability *= self.grams[tag_index]

                token = sequence[index]
                try:
                    token_index = self.word2idx[token]
                except KeyError:
                    token_index = self.word2idx.get('<unk>', None)
                probability *= self.emissions[
                    tag_index * self.n_tokens + token_index]

        elif self.model == 'bigram':
            tags_cleaned = ["<STOP_WORD>"]
            sequence_cleaned = ["<STOP_WORD>"]
            for index, token in enumerate(sequence):
                tags_cleaned.append(tags[index])
                sequence_cleaned.append(token)
                if token in '.!?;' or token == '-docstart-':
                    tags_cleaned.append("<STOP_WORD>")
                    sequence_cleaned.append("<STOP_WORD>")

            if tags_cleaned[-2] != "<STOP_WORD>":
                end_symb = tags_cleaned[-1]
                tags_cleaned[-1] = "."
                tags_cleaned.append("<STOP_WORD>")
                tags_cleaned.append(end_symb)

            if sequence_cleaned[-2] != "<STOP_WORD>":
                end_symb = sequence_cleaned[-1]
                sequence_cleaned[-1] = "."
                sequence_cleaned.append("<STOP_WORD>")
                sequence_cleaned.append(end_symb)

            prev_tag_index = self.tag2idx[tags_cleaned[0]]

            for index, tag in enumerate(tags_cleaned[1:]):
                current_tag_index = self.tag2idx[tag]
                probability *= self.grams[
                    (prev_tag_index * self.n_tags) + current_tag_index]

                token = sequence_cleaned[index + 1]
                try:
                    token_index = self.word2idx[token]
                except KeyError:
                    token_index = self.word2idx.get('<unk>', None)

                probability *= self.emissions[current_tag_index *
                                              self.n_tokens + token_index]

                prev_tag_index = current_tag_index

        elif self.model == 'trigram':
            tags_cleaned = ["<STOP_WORD>", "<STOP_WORD>"]
            sequence_cleaned = ["<STOP_WORD>", "<STOP_WORD>"]
            for index, token in enumerate(sequence):
                tags_cleaned.append(tags[index])
                sequence_cleaned.append(token)
                if token in '.!?;' or token == '-docstart-':
                    tags_cleaned.append("<STOP_WORD>")
                    tags_cleaned.append("<STOP_WORD>")
                    sequence_cleaned.append("<STOP_WORD>")
                    sequence_cleaned.append("<STOP_WORD>")

            if tags_cleaned[-2] != "<STOP_WORD>":
                end_symb = tags_cleaned[-1]
                tags_cleaned[-1] = "."
                tags_cleaned.append("<STOP_WORD>")
                tags_cleaned.append("<STOP_WORD>")
                tags_cleaned.append(end_symb)

            if sequence_cleaned[-2] != "<STOP_WORD>":
                end_symb = sequence_cleaned[-1]
                sequence_cleaned[-1] = "."
                sequence_cleaned.append("<STOP_WORD>")
                sequence_cleaned.append("<STOP_WORD>")
                sequence_cleaned.append(end_symb)

            prev_one_tag_index = self.tag2idx[tags_cleaned[1]]
            prev_two_tag_index = self.tag2idx[tags_cleaned[0]]

            for index, tag in enumerate(tags_cleaned[2:]):
                current_index = self.tag2idx[tag]
                probability *= self.grams[
                    (prev_two_tag_index * (self.n_tags ** 2)) +
                    prev_one_tag_index * self.n_tags + current_index]

                token = sequence_cleaned[index + 2]
                try:
                    token_index = self.word2idx[token]
                except KeyError:
                    token_index = self.word2idx.get('<unk>', None)

                probability *= self.emissions[current_index *
                                              self.n_tokens +
                                              token_index]

                prev_one_tag_index, prev_two_tag_index = current_index, prev_one_tag_index

        return probability

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        Implemented with:
            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        if self.method == 'greedy':
            return self.greedy(sequence)
        elif self.method == 'beam':
            return self.beam(sequence)
        elif self.method == 'viterbi':
            return self.viterbi(sequence)

    def greedy(self, sequence):
        """
        Inference strategy where the max probability for each -gram and emission
        is considered for the prediction.
        """
        output_lst = []
        if self.model == 'unigram':

            prev_token = sequence[0]
            for token in sequence[1:]:
                token_index = self.word2idx.get(token, None)

                if token_index is None:
                    if self.suffix_bool:
                        pred_tag = suffix_groups(token, prev_token)
                        if pred_tag:
                            output_lst.append(pred_tag)
                            prev_token = token
                            continue

                    if not self.unknown_token_handling:
                        # Assign noun
                        output_lst.append('NN')
                        prev_token = token
                        continue

                    else:
                        # convert token to unk
                        token_index = self.word2idx.get('<unk>')

                pred_tag_index = np.argmax(
                    [self.emissions[
                         (tag_index * self.n_tokens) + token_index] *
                     self.grams[tag_index]
                     for tag_index in range(len(self.tag2idx))]
                )

                pred_tag = self.idx2tag[pred_tag_index]

                output_lst.append(pred_tag)

                prev_token = token

        elif self.model == 'bigram':
            seq_cleaned = ["<STOP_WORD>"]
            for token in sequence:
                seq_cleaned.append(token)
                if token in '.!?;' or token == '-docstart-':
                    seq_cleaned.append("<STOP_WORD>")

            remove_bool = False
            if seq_cleaned[-2] != "<STOP_WORD>":
                end_symb = seq_cleaned[-1]
                seq_cleaned[-1] = "."
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append(end_symb)
                remove_bool = True

            prev_tag_index = self.tag2idx[seq_cleaned[0]]
            for token in seq_cleaned[1:]:
                prob_lst = []
                token_index = self.word2idx.get(token, None)

                if token_index is None:
                    if self.suffix_bool:
                        pred_tag = suffix_groups(token, prev_tag_index)
                        if pred_tag:
                            output_lst.append(pred_tag)
                            prev_tag_index = self.tag2idx[pred_tag]
                            continue

                    if not self.unknown_token_handling:
                        # Assign noun
                        output_lst.append('NN')
                        prev_tag_index = self.tag2idx['NN']
                        continue
                    else:
                        # convert token to unk
                        token_index = self.word2idx.get('<unk>')

                elif token == '<STOP_WORD>':
                    prev_tag_index = self.tag2idx['<STOP_WORD>']
                    continue

                for current_tag_index in range(len(self.tag2idx)):
                    prob_lst.append(self.emissions[
                                        current_tag_index * self.n_tokens + token_index] *
                                    self.grams[
                                        prev_tag_index * self.n_tags + current_tag_index])

                pred_tag_index = np.argmax(prob_lst)
                prev_tag_index = pred_tag_index

                pred_tag = self.idx2tag[pred_tag_index]
                output_lst.append(pred_tag)

            if remove_bool:
                val = output_lst.pop(-2)

            if len(output_lst) != len(sequence):
                print("Mismatch Length:", len(output_lst), len(sequence))
                print(output_lst)
                print(sequence)

        elif self.model == 'trigram':
            seq_cleaned = ["<STOP_WORD>", "<STOP_WORD>"]
            for token in sequence:
                seq_cleaned.append(token)
                if token in '.!?;' or token == '-docstart-':
                    seq_cleaned.append("<STOP_WORD>")
                    seq_cleaned.append("<STOP_WORD>")

            remove_bool = False
            if seq_cleaned[-2] != "<STOP_WORD>":
                end_symb = seq_cleaned[-1]
                seq_cleaned[-1] = "."
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append(end_symb)
                remove_bool = True

            prev_two_tag_index = self.tag2idx[seq_cleaned[0]]
            prev_one_tag_index = self.tag2idx[seq_cleaned[1]]
            for token in seq_cleaned[2:]:
                prob_lst = []
                token_index = self.word2idx.get(token, None)

                if token_index is None:
                    if self.suffix_bool:
                        pred_tag = suffix_groups(token, prev_one_tag_index)
                        if pred_tag:
                            output_lst.append(pred_tag)
                            prev_one_tag_index, prev_two_tag_index = \
                                self.tag2idx[pred_tag], prev_one_tag_index
                            continue

                    if not self.unknown_token_handling:
                        # Assign noun
                        output_lst.append('NN')
                        prev_one_tag_index, prev_two_tag_index = \
                            self.tag2idx['NN'], prev_one_tag_index
                        continue
                    else:
                        # convert token to unk
                        token_index = self.word2idx.get('<unk>')

                elif token == '<STOP_WORD>':
                    prev_one_tag_index, prev_two_tag_index = \
                        self.tag2idx['<STOP_WORD>'], prev_one_tag_index
                    continue

                for current_tag_index in range(len(self.tag2idx)):
                    prob_lst.append(self.emissions[
                                        current_tag_index * self.n_tokens + token_index] *
                                    self.grams[(prev_two_tag_index * self.n_tags ** 2) +
                                               prev_one_tag_index * self.n_tags + current_tag_index])

                pred_tag_index = np.argmax(prob_lst)
                prev_one_tag_index, prev_two_tag_index = pred_tag_index, prev_one_tag_index

                pred_tag = self.idx2tag[pred_tag_index]
                output_lst.append(pred_tag)

            if remove_bool:
                val = output_lst.pop(-2)

            if len(output_lst) != len(sequence):
                print("Mismatch Length:", len(output_lst), len(sequence))
                print(output_lst)
                print(sequence)

        return output_lst

    def emission(self, token_idx, tag_idx):
        """
        Vectorized helper function for emission calculation
        """
        temp = self.emissions[tag_idx * self.n_tokens + token_idx]
        if temp == 0:
            return -math.inf
        else:
            return np.log(temp)

    def trans_bigram(self, prev_tag_idx, tag_idx):
        """
        Vectorized helper function for bigram calculation
        """
        temp = self.grams[prev_tag_idx * self.n_tags + tag_idx]

        if temp == 0:
            return -math.inf
        else:
            return np.log(temp)

    def trans_trigram(self, prev_two_index, prev_one_index, tag_idx):
        """
        Vectorized helper function for trigram calculation
        """
        temp = self.grams[(prev_two_index * self.n_tags ** 2) +
                          (prev_one_index * self.n_tags) + tag_idx]

        if temp == 0:
            return -math.inf
        else:
            return np.log(temp)

    def emission_matrix(self):
        token_ar = np.array([[i for _ in range(self.n_tags)] for i in range(self.n_tokens)])
        tag_vec = np.array([_ for _ in range(self.n_tags)])
        vfunc = np.vectorize(self.emission)
        return vfunc(token_ar, tag_vec)

    def find(self, element, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == element:
                    return i, j

    def transition_bi_matrix(self):
        """
        Calculates bigram matrix used for beam search
        """
        tag_ar = np.array([[i for _ in range(self.n_tags)] for i in range(self.n_tags)])
        tag_vec = np.array([_ for _ in range(self.n_tags)])
        vfunc = np.vectorize(self.trans_bigram)
        return vfunc(tag_ar, tag_vec)

    def transition_tri_matrix(self):
        """
        Calculates trigram matrix used for beam search
        """
        tag_ar = np.array([[i for _ in range(self.n_tags)] for i in range(self.n_tags ** 2)])
        tag_vec = np.array([_ for _ in range(self.n_tags)])
        vfunc = np.vectorize(self.trans_trigram)
        return vfunc(tag_ar // self.n_tags, tag_ar % self.n_tags, tag_vec)

    def beam(self, sequence):
        """
        Uses the beam search strategy to infer from the given sequence.
        """
        output_lst = []

        if self.model == 'bigram':
            seq_cleaned = ["<STOP_WORD>"]
            for token in sequence:
                seq_cleaned.append(token)
                if token in '.!?;' or token == '-docstart-':
                    seq_cleaned.append("<STOP_WORD>")

            remove_bool = False
            if seq_cleaned[-2] != "<STOP_WORD>":
                end_symb = seq_cleaned[-1]
                seq_cleaned[-1] = "."
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append(end_symb)
                remove_bool = True

            final_tags = []
            next_prev_tags = [[[self.tag2idx[seq_cleaned[0]]], np.log(1)]]
            for token in seq_cleaned[1:]:
                prev_tags = next_prev_tags
                next_prev_tags = []
                token_index = self.word2idx.get(token, None)

                for prev_tag in prev_tags:

                    if token_index is None:
                        if self.suffix_bool:
                            pred_tag = suffix_groups(token, prev_tag[0][-1])
                            if pred_tag:
                                new_tag_hist = copy.deepcopy(prev_tag[0])
                                new_tag_hist.append(self.tag2idx[pred_tag])
                                trans_p = np.log(self.grams[new_tag_hist[-2] * self.n_tags + self.tag2idx[pred_tag]])
                                new_p = prev_tag[1] + trans_p
                                next_prev_tags.append([new_tag_hist, new_p])
                                continue

                        if not self.unknown_token_handling:
                            # Assign noun
                            pred_tag = self.tag2idx['NN']
                            new_tag_hist = copy.deepcopy(prev_tag[0])
                            new_tag_hist.append(pred_tag)
                            temp = self.grams[new_tag_hist[-2] * self.n_tags + pred_tag]
                            if temp == 0:
                                continue
                            else:
                                trans_p = np.log(temp)
                            new_p = prev_tag[1] + trans_p
                            next_prev_tags.append([new_tag_hist, new_p])
                            continue
                        else:
                            # convert token to unk
                            token_index = self.word2idx.get('<unk>')

                    elif token == '<STOP_WORD>':
                        pred_tag = self.tag2idx['<STOP_WORD>']
                        new_tag_hist = copy.deepcopy(prev_tag[0])
                        new_tag_hist.append(pred_tag)
                        temp = self.grams[new_tag_hist[-2] * self.n_tags + pred_tag]
                        if temp == 0:
                            continue
                        else:
                            trans_p = np.log(temp)
                        new_p = prev_tag[1] + trans_p
                        next_prev_tags.append([new_tag_hist, new_p])
                        continue

                    prev_tag_idx = prev_tag[0][-1]
                    row = self.e_matrix[token_index]
                    col = self.t_matrix[prev_tag_idx, :]
                    vfuc = np.vectorize(lambda x, y: x + y)
                    prob_vec = vfuc(row, col)

                    prob_vec += prev_tag[1]
                    best_idxs = np.argpartition(prob_vec, -self.beam_k)[-self.beam_k:]
                    best_probs = [prob_vec[i] for i in best_idxs]

                    for i in range(self.beam_k):
                        new_tag_hist = copy.deepcopy(prev_tag[0])
                        new_tag_hist.append(best_idxs[i])
                        next_prev_tags.append([new_tag_hist, best_probs[i]])

                next_prev_tags.sort(key=lambda x: x[1], reverse=True)
                pred_tags = next_prev_tags[0:self.beam_k]
                next_prev_tags = pred_tags
                final_tags = next_prev_tags

            final_tags.sort(key=lambda x: x[1], reverse=True)
            final_tag_idxs, probability = final_tags[0][0], final_tags[0][1]
            output_lst = [self.idx2tag[_] for _ in final_tag_idxs]

            idxs = indices(output_lst, '<STOP_WORD>')
            idxs.sort(reverse=True)

            for i in idxs:
                output_lst.pop(i)

            if remove_bool:
                val = output_lst.pop(-2)

            if len(output_lst) != len(sequence):
                print("Mismatch Length:", len(output_lst), len(sequence))
                print(output_lst)
                print(sequence)

        elif self.model == 'trigram':
            seq_cleaned = ["<STOP_WORD>", "<STOP_WORD>"]
            for token in sequence:
                seq_cleaned.append(token)
                if token in '.!?;' or token == '-docstart-':
                    seq_cleaned.append("<STOP_WORD>")
                    seq_cleaned.append("<STOP_WORD>")

            remove_bool = False
            if seq_cleaned[-2] != "<STOP_WORD>":
                end_symb = seq_cleaned[-1]
                seq_cleaned[-1] = "."
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append(end_symb)
                remove_bool = True

            final_tags = []
            final_probs = []
            next_prev_probs = [np.log(1)]
            next_prev_tags = [[self.tag2idx[seq_cleaned[0]], self.tag2idx[seq_cleaned[1]]]]
            for token in seq_cleaned[2:]:
                prev_tags = next_prev_tags
                prev_probs = next_prev_probs
                next_prev_tags = []
                next_prev_probs = []
                token_index = self.word2idx.get(token, None)

                for prev_index, prev_tag in enumerate(prev_tags):

                    if token_index is None:
                        if self.suffix_bool:
                            pred_tag = suffix_groups(token, prev_tag[-1])
                            if pred_tag:
                                pred_tag = self.tag2idx[pred_tag]
                                new_tag_hist = copy.deepcopy(prev_tag)
                                trans_p = np.log(self.grams[(new_tag_hist[-2] * self.n_tags ** 2) + (
                                            new_tag_hist[-1] * self.n_tags) + pred_tag])
                                new_tag_hist.append(pred_tag)
                                next_prev_tags.append(new_tag_hist)
                                new_p = prev_probs[prev_index] + trans_p
                                next_prev_probs.append(new_p)
                                continue

                        if not self.unknown_token_handling:
                            # Assign noun
                            pred_tag = self.tag2idx['NN']
                            new_tag_hist = copy.deepcopy(prev_tag)
                            trans_p = np.log(self.grams[(new_tag_hist[-2] * self.n_tags ** 2) + (
                                        new_tag_hist[-1] * self.n_tags) + pred_tag])
                            new_tag_hist.append(pred_tag)
                            next_prev_tags.append(new_tag_hist)
                            new_p = prev_probs[prev_index] + trans_p
                            next_prev_probs.append(new_p)
                            continue
                        else:
                            # convert token to unk
                            token_index = self.word2idx.get('<unk>')

                    elif token == '<STOP_WORD>':
                        pred_tag = self.tag2idx['<STOP_WORD>']
                        new_tag_hist = copy.deepcopy(prev_tag)
                        trans_p = np.log(self.grams[(new_tag_hist[-2] * self.n_tags ** 2) + (
                                    new_tag_hist[-1] * self.n_tags) + pred_tag])
                        new_tag_hist.append(pred_tag)
                        next_prev_tags.append(new_tag_hist)
                        new_p = prev_probs[prev_index] + trans_p
                        next_prev_probs.append(new_p)
                        continue

                    prev_tag1_idx = prev_tag[-1]
                    prev_tag2_idx = prev_tag[-2]
                    row = self.e_matrix[token_index]
                    col = self.t_matrix[prev_tag2_idx * self.n_tags + prev_tag1_idx, :]
                    vfuc = np.vectorize(lambda x, y: x + y)
                    prob_vec = vfuc(row, col)

                    prob_vec += prev_probs[prev_index]
                    best_idxs = np.argpartition(prob_vec, -self.beam_k)[-self.beam_k:]
                    best_probs = [prob_vec[i] for i in best_idxs]

                    for i in range(self.beam_k):
                        new_tag_hist = copy.deepcopy(prev_tag)
                        new_tag_hist.append(best_idxs[i])
                        next_prev_tags.append(new_tag_hist)
                        next_prev_probs.append(best_probs[i])

                pred_idxs = np.argpartition(next_prev_probs, -self.beam_k)[-self.beam_k:]
                pred_probs = [next_prev_probs[_] for _ in pred_idxs]
                pred_tags = [next_prev_tags[_] for _ in pred_idxs]
                next_prev_probs = pred_probs
                next_prev_tags = pred_tags
                final_probs = next_prev_probs
                final_tags = next_prev_tags

            argmax = np.argmax(final_probs)
            probability = final_probs[argmax]
            final_tag_idxs = final_tags[argmax]
            output_lst = [self.idx2tag[_] for _ in final_tag_idxs]

            idxs = indices(output_lst, '<STOP_WORD>')
            idxs.sort(reverse=True)

            for i in idxs:
                output_lst.pop(i)

            if remove_bool:
                val = output_lst.pop(-2)

            if len(output_lst) != len(sequence):
                print("Mismatch Length:", len(output_lst), len(sequence))
                print(output_lst)
                print(sequence)

        return output_lst

    def viterbi(self, sequence):
        """
        Uses the viterbi stretegy to predict POS tags from given sequence.
        """
        if self.model == 'bigram':

            seq_cleaned = ["<STOP_WORD>"]
            remove_bool = False
            for token in sequence:
                seq_cleaned.append(token)
                if token in '.!?;' or token == '-docstart-':
                    seq_cleaned.append("<STOP_WORD>")

            if seq_cleaned[-2] != "<STOP_WORD>":
                end_symb = seq_cleaned[-1]
                seq_cleaned[-1] = "."
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append(end_symb)
                remove_bool = True

            trellis = np.array([[-math.inf for i in range(len(seq_cleaned))] for j in range(self.n_tags)])
            trellis_p = np.array([[0 for i in range(len(seq_cleaned))] for j in range(self.n_tags)])

            trellis[self.tag2idx['<STOP_WORD>']][0] = np.log(1)
            trellis_p[self.tag2idx['<STOP_WORD>']][0] = 1

            # for each word in the trellis
            prev_token = seq_cleaned[0]
            for col, token in enumerate(seq_cleaned[1:]):
                col += 1
                pred_tag_idx = None
                token_index = self.word2idx.get(token, None)

                # if the word hasn't been seen before
                if token_index is None:
                    # if the word satisfied a suffix rule
                    if self.suffix_bool and suffix_groups(token, prev_token):
                        pred_tag = suffix_groups(token, prev_token)
                        pred_tag_idx = self.tag2idx[pred_tag]

                    elif not self.unknown_token_handling:
                        # Assign noun
                        pred_tag = 'NN'
                        pred_tag_idx = self.tag2idx[pred_tag]

                    # if using <unk> for the unknown word
                    else:
                        # convert token to unk
                        token_index = self.word2idx.get('<unk>')

                # For each current (y_i) POS value
                for row in range(len(trellis)):
                    if pred_tag_idx is not None:
                        if pred_tag_idx == row:
                            emission = 1
                        else:
                            continue
                    else:
                        emission = self.emissions[row * self.n_tokens + token_index]
                    if emission == 0:
                        continue

                    row_probs = []
                    # For each of the previous (y_{i-1}) POS values
                    for row_prev in range(len(trellis)):
                        transition = self.grams[row_prev * self.n_tags + row]
                        if transition == 0:
                            continue
                        row_probs.append(np.log(emission * transition) + trellis[row_prev][col - 1])

                    trellis[row][col] = np.max(row_probs)
                    trellis_p[row][col] = np.argmax(row_probs)
                prev_token = token

            output_lst = []
            row = np.argmax(trellis_p.T[-1])
            output_lst.append('<STOP>')
            for col in range(len(seq_cleaned) - 1, 1, -1):
                row = trellis_p[row][col]
                tag = self.idx2tag[row]
                if tag != '<STOP_WORD>':
                    output_lst.append(tag)

            output_lst.reverse()

            if remove_bool:
                val = output_lst.pop(-2)
                # print('Removed:', seq_cleaned[-3], val)

        if self.model == 'trigram':

            seq_cleaned = ["<STOP_WORD>", "<STOP_WORD>"]
            remove_bool = False
            for token in sequence:
                seq_cleaned.append(token)
                if token in '.!?;' or token == '-docstart-':
                    seq_cleaned.append("<STOP_WORD>")
                    seq_cleaned.append("<STOP_WORD>")

            if seq_cleaned[-2] != "<STOP_WORD>":
                end_symb = seq_cleaned[-1]
                seq_cleaned[-1] = "."
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append("<STOP_WORD>")
                seq_cleaned.append(end_symb)
                remove_bool = True

            trellis = np.array([[-math.inf for _ in range(len(seq_cleaned))] for _ in
                                range(self.n_tags ** 2)])
            trellis_p = np.array([[0 for _ in range(len(seq_cleaned))] for _ in
                                  range(self.n_tags ** 2)])

            trellis[self.tag2idx['<STOP_WORD>'] * self.n_tags + self.tag2idx[
                '<STOP_WORD>']][1] = np.log(1)
            trellis_p[self.tag2idx['<STOP_WORD>'] * self.n_tags + self.tag2idx[
                '<STOP_WORD>']][1] = 1

            trellis[self.tag2idx['<STOP_WORD>'] * self.n_tags + self.tag2idx[
                '<STOP_WORD>']][1] = np.log(1)
            trellis_p[self.tag2idx['<STOP_WORD>'] * self.n_tags + self.tag2idx[
                '<STOP_WORD>']][1] = 1

            trellis[self.tag2idx['<STOP_WORD>'] * self.n_tags + self.tag2idx[
                '<STOP_WORD>']][1] = np.log(1)
            trellis_p[self.tag2idx['<STOP_WORD>'] * self.n_tags + self.tag2idx[
                '<STOP_WORD>']][1] = 1

            # for each word in the trellis
            prev_token = seq_cleaned[1]
            for col, token in enumerate(seq_cleaned[2:]):
                col += 2
                pred_tag_idx = None
                token_index = self.word2idx.get(token, None)

                # if the word hasn't been seen before
                if token_index is None:
                    # if the word satisfied a suffix rule
                    if self.suffix_bool and suffix_groups(token, prev_token):
                        pred_tag = suffix_groups(token, prev_token)
                        pred_tag_idx = self.tag2idx[pred_tag]

                    elif not self.unknown_token_handling:
                        # Assign noun
                        pred_tag = 'NN'
                        pred_tag_idx = self.tag2idx[pred_tag]

                    # if using <unk> for the unknown word
                    else:
                        # convert token to unk
                        token_index = self.word2idx.get('<unk>')

                # For each current (y_i) POS value
                for row in range(len(trellis)):
                    prev_row_fl_loc = row // self.n_tags
                    if pred_tag_idx is not None:
                        if pred_tag_idx == row % self.n_tags:
                            emission = 1
                        else:
                            continue
                    else:
                        emission = self.emissions[(row % self.n_tags) * self.n_tokens + token_index]
                    if emission == 0:
                        continue

                    row_probs = []
                    # For each of the previous (y_{i-1}) POS values
                    for row_prev in range(len(trellis)):
                        if row_prev % self.n_tags != prev_row_fl_loc:
                            continue

                        transition = self.grams[((row_prev // self.n_tags) * self.n_tags ** 2) + (
                                    (row_prev % self.n_tags) * self.n_tags) + (row % self.n_tags)]
                        if transition == 0:
                            continue
                        row_probs.append((row_prev, np.log(emission * transition) + trellis[row_prev][col - 1]))

                    max_tup = max(row_probs, key=lambda x: x[1])
                    trellis[row][col] = max_tup[1]
                    trellis_p[row][col] = max_tup[0]

                prev_token = token

            output_lst = []
            row = np.argmax(trellis_p.T[-1])
            output_lst.append('<STOP>')
            for col in range(len(seq_cleaned) - 1, 1, -1):
                row = trellis_p[row][col]
                tag = self.idx2tag[row // self.n_tags]
                if tag != '<STOP_WORD>':
                    output_lst.append(tag)

            output_lst.reverse()

            if remove_bool:
                val = output_lst.pop(-2)

        if len(output_lst) != len(sequence):
            print("Mismatch Length:", len(output_lst), len(sequence))

        return output_lst


if __name__ == "__main__":
    model = 'trigram'
    method = 'viterbi'
    suffix_bool = True
    unknown_token_handling = True
    smoothing_strategy = 'add-k'
    smooth_k = 0.1
    unk_threshold = 4
    beam_k = 3

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")
    pos_tagger = POSTagger(train_data, unknown_token_handling,
                           smoothing_strategy, model, method,
                           suffix_bool, smooth_k, unk_threshold, beam_k)
    pos_tagger.train()
    evaluate(dev_data, pos_tagger)
