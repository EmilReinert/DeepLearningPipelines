import os
import sys
import re
import ast  # for python AST respresentation
import random
from progress.bar import Bar
import torch
import torch.legacy.nn as nn
# for binary tree LSTM model
import modules.models.e_tree_lstm as etree


def code2ast(code):
    """
    Recursively Parses Code into python AST
    Reference: https://github.com/caterinaurban/Lyra/blob/master/src/lyra/visualization/ast_visualizer.py

    :param code: string representation of to-be-processed code
    :return: dict-AST representation of code
    """
    def transform_ast(code_ast):
        if isinstance(code_ast, ast.AST):
            ast_node = {to_camelcase(k): transform_ast(
                getattr(code_ast, k)) for k in code_ast._fields}
            ast_node['node_type'] = to_camelcase(code_ast.__class__.__name__)
            return ast_node
        elif isinstance(code_ast, list):
            return [transform_ast(el) for el in code_ast]
        else:
            return code_ast
    return transform_ast(ast.parse(code))


def to_camelcase(string):
    """
    converts string to camelcase
    :param string: any string that will be transformed accordingly
    :return: camelcase string of input
    """
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()


def read_file(path):
    """
    reads file and returns it
    :param path: path to 2b-extracted file
    :return: string of file
    """
    with open(path, 'r') as file:
        return file.read()


def tidy(code):
    """
    doing first processes on code before parsing
    :param code: 2b-tidied code
    :return: string code
    """
    # removing comments
    def stripComments(code_str):
        code_str = str(code)
        return re.sub(r'(?m)^ *#.*\n?', '', code_str)

    # removing docstrings TODO
    code = stripComments(code)
    return code


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
                if ast_node in local_dict:
                    local_dict[ast_node] += 1
                else:
                    local_dict[ast_node] = 1
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
                if highest_count < 1:
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
            e_vec.append(random.uniform(-1.0, 1.0))
        e_mat.append(e_vec)
    return e_mat


def ast2vec(ast_node, dictionary, embed_matrix):
    """
    embedding of single ast node

    :param ast_token: 2b-embedded ast  ast_node
    :param dictionary: dictionary of datatokens
    :param embed_matrix: embedding material
    :returns: vector representation of ast
    """
    # find index first
    if ast_node in dictionary:
        index = dictionary.index(ast_node)
    else:
        # last element in dictionary is Unknown type; equals dictionary.index("UNK")
        index = len(dictionary)-1
    # lookup index in embedding matrix
    return embed_matrix[index]


def embed(datasets, dictionary, embed_matrix):
    """
    embedding of ast datasets with dictionary and embedding matrix

    :param dataset: tokenized list, optionally n dimensional
    :param dictionary: dictionary of datatokens
    :param embed_matrix: embedding material
    :returns: vector representation of datafiles
    """
    embedded_datasets = []
    for tree in datasets:
        embedded_data = []
        for ast_node in ast.walk(tree):
            embedded_data.append(ast2vec(ast_node, dictionary, embed_matrix))
        embedded_datasets.append(embedded_data)
    return embedded_datasets


class DefectPrediction:
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
        self.voc_size = 5
        self.vec_length = 3

    def run(self):
        """
        runs whole Pipeline with already initialized defective and clean datasets
        """
    # PREPROCESSING ###########

    # cleaning and opening files TODO manage datacorpus with lables and crawling
        data_def = tidy(read_file(self.raw_data_defective))
        data_cln = tidy(read_file(self.raw_data_clean))
        data_test = tidy(read_file(self.raw_data_test))

    # transforming file strings to AST and filling datasets
        # parsing with own funciton ast_data_def_exp = code2ast(data_def) TODO manage error/empty files
        # parsing with ast.parse
        ast_data_def = []
        ast_data_cln = []
        ast_data_test = []
        ast_data_def.append(ast.parse(data_test))
        ast_data_cln.append(ast.parse(data_cln))
        ast_data_test. append(ast.parse(data_test))

        # prints
        print("Defective Data AST:\n", ast.dump(ast_data_def[0]))
        print("Clean Data AST:\n", ast.dump(ast_data_cln[0]))
        print("Test Data AST:\n", ast.dump(ast_data_test[0]))

    # vocabulary of highest occurences
        self.dictionary = create_dictionary(
            [ast_data_cln, ast_data_def], self.voc_size)  # TODO manage whole datastorage

    # EMBEDDING ########### TODO learning?
        # random embedding of dictionary; as initializing!
        self.emb_matrix = random_embed(self.dictionary, self.vec_length)
        print("Embedding Matrix:\n", self.emb_matrix)

    # NEURAL NETWORK ########### TODO
        # initializing model
        model = NNSimulation(self)

        # training parental predictin on clean data
        model.train_datasets(ast_data_def, ast_data_cln)

    # RESULTS ########### TODO
    # testrun: Prediction
        # obtaining hiddenstate and context of all trees(files) from training data
        # h_root, c_root = DefectPredictor(self.dictionary, self.emb_matrix).predict(ast_data_test)


class NNSimulation:
    """
    The actual NN Module training and testing processes for DefectPrediction.
    with the help of a TreeLSTM it will be able to do defect predictions for one file(one AST)
    """

    def __init__(self, pipeline):
        """
        :param pipeline: holds all necessary information for nnsimulation
        """
        self.dictionary = pipeline.dictionary
        self.emb_dim = len(pipeline.emb_matrix[0])
        self.emb_matrix = nn.LookupTable(len(pipeline.emb_matrix), self.emb_dim)
        self.emb_matrix.weight = pipeline.emb_matrix
        # lstm properties
        # memory dimension
        self.mem_dim = 150
        # learning rate
        self.learning_rate = 0.05
        # word vector embedding learning rate
        self.emb_learning_rate = 0.0
        # minibatch size
        self.batch_size = 25
        # regulation strength
        self.reg = 1e-4
        # simulation module hidden dimension
        self.sim_nhidden = 50

        # optimization configuration
        self.optim_state = {self.learning_rate}

        # negative log likeligood optimization objective
        self.criterion = nn.ClassNLLCriterion()

        # initialize model
        self.etree_lstm = etree.ETreeLSTM(self)
        try:
            self.params, self.grad_params = self.etree_lstm._flatten(
                self.etree_lstm.parameters())
        except:
            self.params = self.grad_params = torch.zeros(1)

    def train_datasets(self, dataset_def, dataset_cln):
        """
        training for the TreeLSTM

        trains upon clean datasets
        TODO training on defective datasets
        :param dataset_def: dataset containing defective asts
        :param dataset_cln: dataset containing clean asts
        """
        self.train_clean(dataset_cln)

    def train_clean(self, trees):
        """
        consists of 3 steps for a tree:
            - recursively (from branch) walk over children and let them predict the parent node
            - Compare the prediction with actual node
            - adjust weights of model so that the difference is minimal
        """
        

    def predict_parent(self, children):
        """
        predicting parent node based on child nodes
        :param children: list of children nodes
        :returns: most likely parent node
        """
        pass

    def predict(self, tree):
        """
        predicting defectiveness of a file/tree
        :param tree: 2b-evaluated abstract sytax tree
        :returns: likelihood of defectiveness 0-1
        """
        pass

    def predict_def_datasets(self, dataset_def, dataset_cln):
        """
        iterates over data and calculates the overall correctness of predictions
        :param dataset_def: dataset containing defective asts
        :param dataset_cln: dataset containing clean asts
        :returns: overall precision of Network 0-1
        """
        pass

############################
    """
    TODO delete the following functions when everything runs
    theyre just here for some lookups but dont have any purpose
    """

    def lstm_unit(self, ast_node, depth=0):
        """
        Process of one LSTM unit.
        Recursively calls learning processes on all children in one tree

        :param ast_node: one Python AST node; First call will be with root Node
        :returns: hidden state and context of node; eventually for the whole AST
        """
        weight = torch.tensor([])  # TODO weights with lstm calculation!!
        w_t = ast2vec(ast_node, self.dictionary,
                      self.emb_matrix)  # embedding of tree
        # sum of children hidden outputs
        h_ = 0
        # child hidden state
        h_k = 0
        # context of child
        c_k = 0
        # forget gates
        f_tk = 0
        # childrem forgetrates times the context
        c_ = 0
        for k in ast.iter_child_nodes(ast_node):
            print(k, depth)
            h_k, c_k = self.lstm_unit(k, depth+1)
            f_tk = torch.nn.Sigmoid()(weight)
            h_ += h_k
            c_ += (f_tk * c_k)
        # input gate
        i_t = torch.nn.Sigmoid()(weight)
        # vector of new candidate values for t
        c_t_ = torch.nn.Tanh()(weight)
        # context
        c_t = i_t * c_t_ + c_
        # output gate
        o_t = torch.nn.Sigmoid()(weight)
        h_t = o_t * torch.nn.Tanh()(c_t)

        return h_t, c_t

    def train_clean_trash(self, trees):
        """
        consists of 3 steps for a tree:
            - recursively (from branch) walk over children and let them predict the parent node
            - Compare the prediction with actual node
            - adjust weights of model so that the difference is minimal
        """
        bar = Bar('Training', max=len(trees))
        self.etree_lstm.train = True
        indices = torch.randperm(len(trees))
        zeros = torch.zeros(self.mem_dim)
        for i in range(1, len(trees)+1, self.batch_size):
            bar.next()  # printing progress
            batch_size = min(i+self.batch_size - 1, len(trees))-i+1

            def f_eval():
                pass

            # torch.optim.Adagrad(self.params, self.optim_state)
        bar.finish()
