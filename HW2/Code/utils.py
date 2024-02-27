import pandas as pd
from tqdm import tqdm 
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from constants import *
# from tagger_constants import *


def unk_replace(sentences, values):
    for index in range(len(sentences)):
        for val in values:
            locations = indices(sentences[index], val)
            for loc in locations:
                sentences[index][loc] = '<unk>'
    return sentences


def infer_sentences(model, sentences, start):
    """

    Args:
        model_name: type of model.
        model (POSTagger): model used for inference
        sentences (list[str]): list of sentences to infer by single process
        start (int): index of first sentence in sentences in the original list of sentences

    Returns:
        dict: index, predicted tags for each sentence in sentences
    """
    res = {}
    for i in tqdm(range(len(sentences))):
        res[start+i] = model.inference(sentences[i])
    return res


def compute_prob(model, sentences, tags, start):
    """

    Args:
        model (POSTagger): model used for inference
        sentences (list[str]): list of sentences 
        tags (list[str]): list of tags
        start (int): index of first sentence in sentences in the original list of sentences


    Returns:
        dict: index, probability for each sentence,tag pair
    """
    res = {}
    for i in range(len(sentences)):
        res[start+i] = model.sequence_probability(sentences[i], tags[i])
    return res
    

#from https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list    
def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    df_sentences = pd.read_csv(open(sentence_file))
    doc_start_indexes = df_sentences.index[df_sentences['word'] == '-DOCSTART-'].tolist()
    num_sentences = len(doc_start_indexes)

    sentences = [] # each sentence is a list of tuples (index,word)
    if tag_file:
        df_tags = pd.read_csv(open(tag_file))
        tags = []
    for i in tqdm(range(num_sentences)):
        index = doc_start_indexes[i]
        if i == num_sentences-1:
            # handle last sentence
            next_index = len(df_sentences)
        else:
            next_index = doc_start_indexes[i+1]

        sent = []
        tag = []
        for j in range(index, next_index):
            word = str(df_sentences['word'][j]).strip()
            if not CAPITALIZATION or word == '-DOCSTART-':
                word = word.lower()
            sent.append(word)
            if tag_file:
                tag.append((df_tags['tag'][j]))
        if STOP_WORD:
            sent.append('<STOP>')
        sentences.append(sent)
        if tag_file:
            if STOP_WORD:
                tag.append('<STOP>')
            tags.append(tag)

    if tag_file:
        return sentences, tags

    return sentences

def confusion_matrix(tag2idx,idx2tag, pred, gt, fname):
    """Saves the confusion matrix

    Args:
        tag2idx (dict): tag to index dictionary
        idx2tag (dict): index to tag dictionary
        pred (list[list[str]]): list of predicted tags
        gt (_type_): _description_
        fname (str): filename to save confusion matrix

    """

    flat_pred = list(pred)
    flat_y = list(gt)
    matrix = np.zeros((len(flat_pred), len(flat_y)))
    for i in range(len(flat_pred)):
        idx_pred = tag2idx[flat_pred[i]]
        idx_y = tag2idx[flat_y[i]]
        matrix[idx_y][idx_pred] += 1
    df_cm = pd.DataFrame(matrix, index=[idx2tag[i] for i in range(len(tag2idx))],
                columns=[idx2tag[i] for i in range(len(tag2idx))])
    df_cm = df_cm.loc[["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]

    plt.figure(figsize=(10,7))
    sn.heatmap(df_cm, annot=False, vmax=1000, cmap="Blues")
    plt.savefig(fname)


def write_csv(data, filename):
    df = pd.DataFrame(data).rename(columns={0: "tag"})
    df = df[df['tag'] != '<STOP>']
    df.insert(0, 'id', range(0, len(df)))
    df.to_csv(filename, index=False)
