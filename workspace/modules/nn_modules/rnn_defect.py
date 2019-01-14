import os
import sys
import re
import ast  # for python AST respresentation
import random
import torch
import torch.legacy.nn as nn
import modules.models.rnn_example as rnn


class RNNDefect():
    """
    RNN Neural network for processes of Defect prediction.
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

        # rnn properties
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

        # models
        # initialize code learning model
        self.rnn = rnn.RNN_Example(self)
        # initialize classification model

    def train_datasets(self, dataset_def, dataset_cln, test_cln):
        """
        training for defect prediction

        ! consists of 2 main training steps:
        1 training RNN to learn how code should look like
        2 training a classifier with labeled(clean/defective) code

        :param dataset_def: dataset containing defective asts
        :param dataset_cln: dataset containing clean asts
        :param test_cln: dataset containing clean test asts
        """
        # training rnn
        self.train_clean(dataset_cln, test_cln)
        # training classifier
        self.train_pred(dataset_def, dataset_cln)

    def predict(self, test):
        """
        calls testdata upon RNN to obtain accuracy.
        the accuracy will be classified to find out how defective files can be
        :param test: testdata whose defectiveness is tested
        :returns: true if likely to be defective; false if not
        """

        def classify(accuracy):
            """
            classification process to determine the probability of data based on
            NN code recunstruction accuracy
            :param accuracy: NN code recunstruction accuracy
            :returns: percentage of likelihood of defectiveness
            """
            return 1-accuracy

        accuracy = self.rnn.test_network(test)# TODO couldnt ork
        bug_prob = classify(accuracy)
        if bug_prob > 0.5:
            return 1
        else:
            return 0

############################### TRAINIGS #############################

    # Training of RNN #

    def train_clean(self, dataset_cln, test_cln):
        """
        trains and tests RNN with clean datafiles

        :param dataset_cln: dataset containing clean training asts
        :param test_cln: dataset containing clean test asts
        :returns: void; saves best model in folder

        TODO delete/adjust when made use of TreeLSTM
        """
        # preparing in and outputs for recurrent neural network
        train_in_out = self.prepare_parent_children(dataset_cln)
        test_in_out = self.prepare_parent_children(test_cln)
        print("Training OUT<->IN:\n", train_in_out)
        # print("Testin IN OUT:\n", test_in_out)

        # embedding of datasets | Makes only sense for RNN because we loose context
        emb_train = self.embed(train_in_out)
        # print("Embedded Training IN OUT\n", emb_train)
        emb_test = self.embed(test_in_out)
        self.rnn.run(emb_train, emb_test)

    def prepare_parent_children(self, datasets):
        """
        creates all traing/testing/validation data in and outputs for RNN
        :param datasets: list containing ASTs
        :retuns: 2 dimensional list that holds list of all children for parent at same index
                -> for all datasets
        TODO this works without lstmcontext now because we use standard RNN
            that must be changed/deleted later and be processed in the tree LSTM
        """
        
        # collect parent children pairs first
        all_parents = []
        all_children = []
        for tree in datasets:
            parents, chilren = self.parents_children(tree, self.in_len)
            all_parents.append(parents)
            all_children.extend(chilren)
        return [all_parents, all_children]

    def parents_children(self, tree, sequencelength):
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

                    # im sorry for everyone who has to see this
                    while len(loc_children) < sequencelength:
                        loc_children.append("")

                    children.append(loc_children)

            return [parents, children]

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
            parents.append(self.ast2vec(parent))
        # print(parents)
        embedded_datasets.append(parents)
        all_children = []
        for children in datasets[1]:
            embedded_children = []
            for child in children:
                # creating combined vector of children vec values
                # im sorry for everyone who has to see this code
                embedded_children.append(self.ast2vec(child))
            all_children.append(embedded_children)
        embedded_datasets.append(all_children)
        return embedded_datasets

    def ast2vec(self, ast_node):
        """
        embedding of single ast node

        :param ast_token: 2b-embedded ast  ast_node
        :param dictionary: dictionary of datatokens
        :param embed_matrix: embedding material
        :returns: vector representation of ast
        """
        # find index first
        if ast_node in self.dictionary:
            index = self.dictionary.index(ast_node)
        else:
            # last element in dictionary is Unknown type; equals dictionary.index("UNK")
            index = len(self.dictionary)-1
        # lookup index in embedding matrix
        return self.emb_matrix[index]

    #Training the classification #

    def train_pred(self, def_data, cln_data):
        """
        trains module's classifier with 2 types of data
        :param def_data: defective labled data
        :param cln_data: clean labled data
        :returns: void; saves best model in folder
        """
        # list containing accuracies as inputs for log regression
        cln_in_out = self.prepare_parent_children(cln_data)

