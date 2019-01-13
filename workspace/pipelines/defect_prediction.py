import os
import sys
import re
import ast  # for python AST respresentation
import random
from progress.bar import Bar
import torch
import torch.legacy.nn as nn
# for childsum tree LSTM model
import modules.nn_modules.tree_lstm_defect as tlstm_dp
import modules.nn_modules.rnn_defect as rnn_dp


def read_file(path):
    """
    reads file and returns it
    :param path: path to 2b-extracted file
    :return: string of file
    """
    with open(path, 'r') as file:
        return file.read()


def tidy(files):
    """
    doing first tidying processes on code before parsing
    :param files: 2b-tidied code files
    :returns: string code
    """
    # removing comments
    def stripComments(code_str):
        code_str = str(code)
        return re.sub(r'(?m)^ *#.*\n?', '', code_str)

    # removing docstrings TODO

    t_files = []
    for code in files:
        t_files.append(stripComments(code))
    return t_files


def parse_data(data):
    """
    parsing process for python ast representation
    :param data: already tidied code files
    :returns: list containing all parsed files 
    """
    ast_data = []
    for code in data:
        ast_data.append(ast.parse(code))
    return ast_data


def create_dictionary(datasets, max_count):
    """
    Builds a fixed sized dictionary, with an <UNK> entry at last position

    :param dataset: tokenized list, optionally n dimensional
    :param max_count: defines how long our dictionary will be
    :return: top 'max_count' tokens of dictionary as a list
    """

    def extract_highest_occurences(dataset, max_count):
        """
        finds highest token occurences in datasets

        :param dataset: tokenized list, optionally n dimensional
        :param max_count: defines how long our dictionary will be
        :return: top 'max_count' tokens of dictionary as a list
        """
        # global fixed size dictionary as set
        global_dict = {}
        # iterating over all training files
        for tree in dataset:
            # a complete dictionary for one file
            local_dict = {}
            # iterating over all words of a file
            for ast_node in ast.walk(tree):
                # if word not in the local dictionary then we add it
                # otherwise we rise count
                node = ast_node.__class__.__name__
                if node in local_dict:
                    local_dict[node] += 1
                else:
                    local_dict[node] = 1
            # local dict counts will now be merged into fix sized, global dictionary
            for ast_node in local_dict:
                if ast_node in global_dict:
                    global_dict[ast_node] += local_dict[ast_node]
                else:
                    global_dict[ast_node] = local_dict[ast_node]

            # global dict will be filled with highest counts
            # first we find highest count
            highest_count = 0
            for ast_node in global_dict:
                if highest_count < global_dict[ast_node]:
                    highest_count = global_dict[ast_node]
            # now we create an updated highest count global dict
            new_global_dict = {}
            while len(new_global_dict) < max_count:
                if highest_count < 1 or len(new_global_dict) >= 2*len(global_dict):
                    break
                # filling new global
                for ast_node in global_dict:
                    if (global_dict[ast_node] == highest_count and
                            len(new_global_dict) < max_count):
                        new_global_dict[ast_node] = highest_count
                highest_count -= 1
            global_dict = new_global_dict
        # print("maxcout", global_dict)
        return list(global_dict)

    # running through dimensions
    for dataset in datasets:
        dictionary = extract_highest_occurences(dataset, max_count-1)
        dictionary.append("UNK")
    return dictionary


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])


def random_embed(dictionary, vector_length):
    """
    embeds dictionary with random valued vectors for initializing random weights.
    values lay between -1 and 1

    :param dictionary: 2b-embedded dictionary
    :param vector_length: desired length for vectors
    :returns: embedded dictionary as matrix
    """
    e_mat = []
    # create matric with dimension dictionarysize x vector length
    for mi in range(len(dictionary)):
        e_vec = []
        for vi in range(vector_length):
            e_vec.append(truncate(random.uniform(-1.0, 1.0), 2))
        e_mat.append(e_vec)
    return e_mat


class DefectPrediction:  # main pipeline
    """
    implementation Attemt to a published Paper:
    'A deep tree-based model for software defect prediction'
    Reference: https://arxiv.org/abs/1802.00921

    TASK: Predicting Probability of a Code Being Defective or not
    """

    def __init__(self, data_defective, data_clean, data_test):
        """
        :param data_defective: path to datacorpus code labled as defective
        :param data_clean: path to datacorpus code labled as clean
        """
        self.raw_data_defective = data_defective
        self.raw_data_clean = data_clean
        self.raw_data_test = data_test
        # vocabulary/dictionary size
        self.voc_size = 100
        self.vec_length = 3

    def run(self):
        """
        runs whole Pipeline with already initialized defective and clean datasets
        """
    # PREPROCESSING ###########

        # cleaning and opening files TODO manage datacorpus with lables and crawling
        data_def = tidy([read_file(self.raw_data_defective)])
        data_cln = tidy([read_file(self.raw_data_clean)])
        data_test = tidy([read_file(self.raw_data_test)])

        # transforming file strings to AST and filling datasets
        # parsing with own funciton ast_data_def_exp = code2ast(data_def) TODO manage error/empty files
        # parsing with ast.parse

        ast_data_def = parse_data(data_def)
        ast_data_cln = parse_data(data_cln)
        ast_data_test = parse_data(data_test)

        # print ast data
        #("Defective Data AST:\n", ast.dump(ast_data_def[0]))
        #print("Clean Data AST:\n", ast.dump(ast_data_cln[0]))
        print("Test Data AST:\n", ast.dump(ast_data_test[0]))

        # vocabulary of highest occurences
        self.dictionary = create_dictionary(
            [ast_data_cln, ast_data_def], self.voc_size)  # TODO manage whole datastorage
        print("Dictionary:\n", self.dictionary)

    # EMBEDDING ########### TODO learning?
        # random embedding of dictionary; as initializing!
        self.emb_matrix = random_embed(self.dictionary, self.vec_length)
        # print("Embedding Matrix:\n", self.emb_matrix)

    # NEURAL NETWORK ########### TODO
        # initializing model
        model = rnn_dp.RNNDefect(self)

        # training parental predictin on clean data (test is also used for general testing)
        model.train_datasets(ast_data_def, ast_data_cln, ast_data_test)

        # predicting defectiveness of test data
    # RESULTS ###########
        defective = model.predict(ast_data_test)
        if(defective):
            print("The Test Data is likely to be defective")
        else:
            print("The Test Data is likely to be not defective")
