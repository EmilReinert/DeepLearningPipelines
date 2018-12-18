import os
import sys
import re
import ast  # for python ast respresentation


def code2ast(code):
    """
    Recursively Parses Code into python AST
    Reference: https://github.com/caterinaurban/Lyra/blob/master/src/lyra/visualization/ast_visualizer.py

    :param code: string representation of to-be-processed code
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
    """
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()


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

    def run(self):
        """
        runs whole Pipeline with already initialized defective and clean datasets
        """
        ########### PREPROCESSING ###########

        # transforming all data to AST
        ast_data_def = code2ast(self.data_defective)
        ast_data_clean = code2ast(self.data_clean)

        ########### EMBEDDING ########### TODO


        ########### NEURAL NETWORK ########### TODO


        ########### RESULTS ########### TODO


