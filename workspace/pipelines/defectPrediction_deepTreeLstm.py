import os
import sys
import re
import ast  # for python AST respresentation
import random
from progress.bar import Bar
import torch
import torch.legacy.nn as nn
# for childsum tree LSTM model
import modules.models.e_tree_lstm as etree
import modules.models.rnn_example as rnn


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
        data_def = tidy(read_file(self.raw_data_defective))
        data_cln = tidy(read_file(self.raw_data_clean))
        data_test = tidy(read_file(self.raw_data_test))

        # transforming file strings to AST and filling datasets
        # parsing with own funciton ast_data_def_exp = code2ast(data_def) TODO manage error/empty files
        # parsing with ast.parse
        ast_data_def = []
        ast_data_cln = []
        ast_data_test = []
        ast_data_def.append(ast.parse(data_def))
        ast_data_cln.append(ast.parse(data_cln))
        ast_data_test. append(ast.parse(data_test))

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
        model = RNNDefect(self)

        # training parental predictin on clean data (test is also used for general testing)
        model.train_datasets(ast_data_def, ast_data_cln, ast_data_test)

        # predicting defectiveness of test data
    # RESULTS ########### TODO
    # testrun: Prediction
        # obtaining hiddenstate and context of all trees(files) from training data
        # h_root, c_root = DefectPredictor(self.dictionary, self.emb_matrix).predict(ast_data_test)


class RNNDefect():
    """
    temporary RNN Neural network for processes of Defect prediction.
    Its use is to understand the core processes of the Defect prediction Task
    -> that is archieved by adjusting the processes that are meant for the treelstm
    """

    def __init__(self, pipeline):
        """
        :param pipeline: holds all necessary information for nnsimulation
        """
        # global dictionary
        self.dictionary = pipeline.dictionary
        # vector length
        self.emb_dim = len(pipeline.emb_matrix[0])
        self.emb_matrix = pipeline.emb_matrix
        # embedding matrix as lookup table
        self.emb_matrix_look = nn.LookupTable(
            len(pipeline.emb_matrix), self.emb_dim)
        self.emb_matrix_look.weight = self.emb_matrix
        # length of input sequence for RNN
        self.in_len = 3

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
        self.lstm = rnn.RNN_Example(self)

    def train_datasets(self, dataset_def, dataset_cln, test_cln):
        """
        training for RNN
        trains upon clean datasets
        TODO training on defective datasets?
        :param dataset_def: dataset containing defective asts
        :param dataset_cln: dataset containing clean asts
        :param test_cln: dataset containing clean test asts
        """
        self.train_clean(dataset_cln, test_cln)

    def train_clean(self, dataset_cln, test_cln):
        """
        trains and tests RNN with clean datafiles

        :param dataset_cln: dataset containing clean training asts
        :param test_cln: dataset containing clean test asts
        :returns: void; saves best model in folder

        TODO delete/adjust when made use of TreeLSTM
        """
        # preparing in and outputs for neural network
        train_in_out = self.prepare_in_out(dataset_cln)
        test_in_out = self.prepare_in_out(test_cln)
        print("Training IN OUT:\n", train_in_out)
        # print("Testin IN OUT:\n", test_in_out)

        # embedding of datasets | Makes only sense for RNN because we loose context
        emb_train = self.embed(train_in_out)
        # print("Embedded Training IN OUT\n", emb_train)
        emb_test = self.embed(test_in_out)
        self.lstm.run(emb_train, emb_test)

    def embed(self, datasets):
        """
        embedding for 2 dimensional list containing ast node names
        of parents in [0] and children in [1]

        :param dataset: tokenized list, optionally n dimensional
        :param dictionary: dictionary of datatokens
        :param embed_matrix: embedding material
        :returns: vector representation of datafiles
        """
        embedded_datasets = []
        parents = []
        for parent in datasets[0]:
            parents.append(ast2vec(
                parent, self.dictionary, self.emb_matrix))
        # print(parents)
        embedded_datasets.append(parents)
        all_children = []
        for children in datasets[1]:
            embedded_children = []
            for child in children:
                # creating combined vector of children vec values
                # im sorry for everyone who has to see this code
                embedded_children.append(ast2vec(
                    child, self.dictionary, self.emb_matrix))
            all_children.append(embedded_children)
        embedded_datasets.append(all_children)
        return embedded_datasets

    def prepare_in_out(self, datasets):
        """
        creates all traing/testing/validation data in and outputs for NN
        :param datasets: list containing ASTs
        :retuns: 2 dimensional list that holds list of all children for parent at same index
                -> for all datasets
        TODO this works without lstmcontext now because we use standard RNN
            that must be changed/deleted later and be processed in the tree LSTM
        """
        def parents_children(tree, sequencelength):
            """
            extracts all parents with its children from given AST
            :param tree: 2b-extracted python AST
            :returns: 2 lists that represent list of all children for parent at same index
            """
            parents = []
            children = []
            for node in ast.walk(tree):
                loc_children = []
                loc_children_ast = ast.iter_child_nodes(node)
                # test if node is branch, if yes then its ignored
                for child_ast in loc_children_ast:
                    loc_children.append(child_ast.__class__.__name__)
                if not len(loc_children) == 0:
                    parents.append(node.__class__.__name__)

                    # make children list sized 5 so it works with lstm fix input length
                    # im sorry for everyone who has to see this
                    while len(loc_children) < sequencelength:
                        loc_children.append("")

                    children.append(loc_children)

            return [parents, children]
        # collect parent children pairs first
        all_parents = []
        all_children = []
        for tree in datasets:
            parents, chilren = parents_children(tree, self.in_len)
            all_parents.extend(parents)
            all_children.extend(chilren)
        return [all_parents, all_children]


class TreeLSTMDefect:
    """
    The actual TreeLSTM Module training and testing processes for DefectPrediction.
    with the help of a TreeLSTM it will be able to do defect predictions for one file(one AST)
    """

    def __init__(self, pipeline):
        """
        :param pipeline: holds all necessary information for nnsimulation
        """
        # global dictionary
        self.dictionary = pipeline.dictionary
        # vector length
        self.emb_dim = len(pipeline.emb_matrix[0])
        self.emb_matrix = pipeline.emb_matrix
        # embedding matrix as lookup table
        self.emb_matrix_look = nn.LookupTable(
            len(pipeline.emb_matrix), self.emb_dim)
        self.emb_matrix_look.weight = self.emb_matrix
        # length of input sequence for RNN
        self.in_len = 3

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

        '''
        self.etree_lstm = etree.ETreeLSTM(self)
        try:
            self.params, self.grad_params = self.etree_lstm._flatten(
                self.etree_lstm.parameters())
        except:
            self.params = self.grad_params = torch.zeros(1)
        '''

    def train_datasets(self, dataset_def, dataset_cln, test_cln):
        """
        training for the TreeLSTM
        """

    def train_clean(self, dataset_cln, test_cln):
        """
        trains and tests TreeLSTM with clean datafiles
        TODO in TreeLSTM
        consists of 3 steps for a tree:
            - recursively (from branch) walk over children and let them predict the parent node
            - Compare the prediction with actual node
            - adjust weights of model so that the difference is minimal

        :param dataset_cln: dataset containing clean training asts
        :param test_cln: dataset containing clean test asts
        :returns: void; saves best model in folder
        """

    def predict_parent(self, children):
        """
        predicting parent node based on child nodes
        : param children: list of children nodes
        : returns: most likely parent node
        """
        pass

    def predict(self, tree):
        """
        predicting defectiveness of a file/tree
        : param tree: 2b-evaluated abstract sytax tree
        : returns: likelihood of defectiveness 0-1
        """
        pass

    def predict_def_datasets(self, dataset_def, dataset_cln):
        """
        iterates over data and calculates the overall correctness of predictions
        : param dataset_def: dataset containing defective asts
        : param dataset_cln: dataset containing clean asts
        : returns: overall precision of Network 0-1
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

        : param ast_node: one Python AST node; First call will be with root Node
        : returns: hidden state and context of node; eventually for the whole AST
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
            - recursively(from branch) walk over children and let them predict the parent node
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
