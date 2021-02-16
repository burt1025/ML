import numpy as np
from hmm import HMM

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)
    ###################################################
    # TODO: build two dictionaries
    #   - from a word to its index 
    #   - from a tag to its index 
    # The order you index the word/tag does not matter, 
    # as long as the indices are 0, 1, 2, ...
    ###################################################
    num_line = len(train_data)
    word2idx = {word : idx for idx, word in enumerate(unique_words.keys())} 
    tag2idx = {tag : idx for idx, tag in enumerate(tags)}

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))
    ###################################################
    # TODO: estimate pi, A, B from the training data.
    #   When estimating the entries of A and B, if  
    #   "divided by zero" is encountered, set the entry 
    #   to be zero.
    ###################################################
    
    for line in train_data:
        pi[tag2idx[line.tags[0]]] += 1
        for i in range(line.length - 1):
            A[tag2idx[line.tags[i]], tag2idx[line.tags[i + 1]]] += 1
        for i in range(line.length):
            B[tag2idx[line.tags[i]], word2idx[line.words[i]]] += 1
            
    pi = pi / num_line
    for i in range(S):
        if (sum(A[i, :]) != 0):
            A[i, :] /= sum(A[i, :])

        if (sum(B[i, :]) != 0):
            B[i, :] /= sum(B[i, :])

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ######################################################################
    # TODO: for each sentence, find its tagging using Viterbi algorithm.
    #    Note that when encountering an unseen word not in the HMM model,
    #    you need to add this word to model.obs_dict, and expand model.B
    #    accordingly with value 1e-6.
    ######################################################################
    S = len(tags)
    for line in test_data:
        for word, tag in zip(line.words, line.tags):
            if word not in model.obs_dict.keys():
                model.obs_dict[word] = len(model.obs_dict)
                model.B = np.hstack((model.B,np.zeros((model.B.shape[0],1))))
                model.B[model.state_dict[tag], model.obs_dict[word]] = 1e-6
        tagging.append(model.viterbi(line.words))
    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
