import os
import sys
import re
import ast  # for python AST respresentation


def code2ast(code):
    """
    Recursively Parses Code into python AST
    Reference: https://github.com/caterinaurban/Lyra/blob/master/src/lyra/visualization/ast_visualizer.py

    :param code: string representation of to-be-processed code
    :return: dict-AST representation of code
    """
    def transform_ast(code_ast):
        if isinstance(code_ast, ast.AST):
            node = {to_camelcase(k): transform_ast(
                getattr(code_ast, k)) for k in code_ast._fields}
            node['node_type'] = to_camelcase(code_ast.__class__.__name__)
            return node
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


def generate_ast_dict(root):
    """
    extracts a
    """
    pass


def extract_highest_occurences(dataset, max_count):
    """
    Builds a fixed sized dictionary by iterating over dataset.

    :param dataset: tokenized list, optionally n dimensional
    :param max_count: defines how long our dictionary will be
    :return: top 'max_count' tokens of dictionary as a list
    """
    # global fixed size dictionary as set
    global_dict = {}
    # iterating over all training files
    for data in dataset:
        # a complete dictionary for one file
        local_dict = {}
        # iterating over all words of a file
        for w in ast.walk(data):
            # if word not in the local dictionary then we add it
            # otherwise we rise count
            if w in local_dict:
                local_dict[w] += 1
            else:
                local_dict[w] = 1
        # local dict counts will now be merged into fix sized, global dictionary
        for w in local_dict:
            if w in global_dict:
                global_dict[w] += local_dict[w]
            else:
                global_dict[w] = local_dict[w]

        # global dict will be filled with highest counts
        # first we find highest count
        highest_count = 0
        for w in global_dict:
            if highest_count < global_dict[w]:
                highest_count = global_dict[w]
        # now we create an updated highest count global dict
        new_global_dict = {}
        while len(new_global_dict) < max_count:
            if highest_count < 1:
                break
            # filling new global
            for w in global_dict:
                if (global_dict[w] == highest_count and
                        len(new_global_dict) < max_count):
                    new_global_dict[w] = highest_count
            highest_count -= 1
        global_dict = new_global_dict
    # print("maxcout", global_dict)
    return list(global_dict)


class DefectPredictor:
    """
    implementation Attemt to a published Paper: 
    'A deep tree-based model for software defect prediction' 
    Reference: https://arxiv.org/abs/1802.00921

    TASK: Predicting Probability of a Code Being Defective or not
    """

    def __init__(self, data_defective, data_clean):
        """
        :param data_defective: path to datacorpus code labled as defective
        :param data_clean: path to datacorpus code labled as clean
        """
        self.data_defective = data_defective
        self.data_clean = data_clean
        self.voc_size = 5

    def run(self):
        """
        runs whole Pipeline with already initialized defective and clean datasets
        """
        ########### PREPROCESSING ###########

        # cleaning and opening files
        data_def = tidy(read_file(self.data_defective))
        data_cln = tidy(read_file(self.data_clean))

        # transforming file strings to AST
        # parsing with own funciton ast_data_def_exp = code2ast(data_def) TODO manage error/empty files
        # parsing with ast.parse
        ast_data_def = ast.parse(data_def)
        ast_data_cln = ast.parse(data_cln)

        # prints
        print(ast.dump(ast_data_def))
        print(ast.dump(ast_data_cln))

        # vocabulary of highest occurences
        self.dictionary = extract_highest_occurences(
            [ast_data_cln, ast_data_def], self.voc_size)  # TODO manage whole datastorage TODO unkown datatype?

        # EMBEDDING ########### TODO

        # NEURAL NETWORK ########### TODO

        # RESULTS ########### TODO
