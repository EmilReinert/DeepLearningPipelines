import torch.nn as nn
from torch.utils.data.dataset import Dataset
import numpy as np
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class RNN_Example():
    """
    Implementation of Recurrent Neural Network.
    For Emils Pipeline Implementation
    """

    def __init__(self, config):
        self.dict = config.dictionary
        self.batch_size = config.batch_size
        self.epochs = 500
        self.input_length = 10
        self.output_length = 1
        self.dict_size = len(self.dict)
        self.saved_path = "/home/emil/Documents/DeepLearningProject/PaperImplementation/DeepLearningPipelines/workspace/dump"
        self.saved_file = os.path.join(self.saved_path, "best_trained_model")
        # TODO: current model not taking sequences of token, only 1 token

    def run(self, train_in, train_out):

        self.train_network(train_in, train_out)
        return list(self.predict_testing_output.numpy())

    def train_network(self, train_in, train_out):
        """
        Train network.

        Train each epoch by training set, evaluate model after each epoch
        using validation set, save the best model and test using test set.

        This function also prints out loss, accuracy each epoch
        and loss/accuracy of the best model.

        :param datasets: list of input/output sets,
        :returns: none
        """

        training_input = train_in
        training_output = train_out
        validating_input = train_in
        validating_output = train_out

        # Used to compare with accuracy of model
        best_accuracy = 0.0

        params = {
            "batch_size":   self.batch_size,
            "shuffle": True,
            "drop_last": True
        }

        # Datasets object generate data which will put into neural network
        # Datasets contain some specific functions to adapt nn in Pytorch
        train_data = Datasets(training_input, training_output)
        valid_data = Datasets(validating_input, validating_output)

        # DataLoader used to load data equal to batch_size
        train_loader = DataLoader(train_data, **params)
        valid_loader = DataLoader(valid_data, **params)

        model = RNN(training_data=train_in, dict_size=self.dict_size)

        # Check if computer have graphic card,
        # model will be trained py GPU instead of CPU
        if torch.cuda.is_available():
            model.cuda()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimization
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Number of iteration ( = length of data / batch_size)
        self.num_iter = int(train_data.__len__()/self.batch_size)

        for epoch in range(self.epochs):
            # Declare to start training phase
            model.train()
            for iter, (content, label) in enumerate(train_loader):
                start_time = time.time()
                if torch.cuda.is_available():
                    content = content.cuda()
                    label = label.cuda()
                # Clean buffer to avoid accumulate value of buffer
                optimizer.zero_grad()

                # Training model to understand the content of traning set
                predicted_value = model(content)

                # Calculating loss
                loss = self.criterion(predicted_value, label)

                # Back propagation
                loss.backward()

                # Optimizing model based on loss
                optimizer.step()
                elapse_time = time.time() - start_time

            # validate this model at the end of each epoch
            self.access_model(
                model=model,
                data_loader=valid_loader,
                access_data=valid_data,
                criterion=self.criterion,
                num_iter=self.num_iter,
                epoch=epoch,
                best_accuracy=best_accuracy)
        
        self.test_network(train_in, train_out)

    def test_network(self, test_in, test_out):
        """
        testing of pretrained network
        :param test_data: data 2b tested with current/best model
        :returns: accuracy
        """
        testing_input = test_in
        testing_output = test_out
        params = {
            "batch_size":   self.batch_size,
            "shuffle": True,
            "drop_last": True
        }
        test_data = Datasets(testing_input, testing_output)
        test_loader = DataLoader(test_data, **params)
        model = RNN([], dict_size=self.dict_size)
        model.load_state_dict(torch.load(self.saved_file))
        accuracy = self.access_model(model=model,
                                     data_loader=test_loader,
                                     access_data=test_data,
                                     criterion=self.criterion,
                                     mode="test",
                                     num_iter=self.num_iter)
        return np.around(accuracy, decimals=3)

    def access_model(self, model, data_loader, access_data, criterion,
                     num_iter, mode="validate", epoch=0, best_accuracy=0.0):
        """
        Validate model after every epoch

        :param model: TODO @Annie @Thang
        :param data_loader: TODO @Annie @Thang
        :param access_data: TODO @Annie @Thang
        :param criterion: TODO @Annie @Thang
        :param num_iter: integer
        :param mode: string TODO @Annie @Thang
        :param epoch: integer
        :param best_accuracy: float
        """
        # Declare to start validating phase
        model.eval()
        loss_list = []
        accuracy_list = []

        if mode == "test":
            self.predict_testing_output = torch.LongTensor([])
        for iter, (content, label) in enumerate(data_loader):
            if torch.cuda.is_available():
                content = content.cuda()
                label = label.cuda()
            # In testing phase, we don't optimize model,
            # we only use model to predict value in testing set
            with torch.no_grad():
                predicted_value = model(content)
                prediction = torch.argmax(predicted_value, dim=1)

                if mode == "test":
                    self.predict_testing_output = torch.cat(
                        (self.predict_testing_output, prediction))

            # Comparing between truth output and predicted output
            accuracy = get_accuracy(prediction=prediction,
                                    actual_value=label,
                                    dict=self.dict)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                if mode == "validate":
                    torch.save(model.state_dict(), self.saved_file)

            loss = criterion(predicted_value, label)
            loss_list.append(loss * label.size()[0])
            accuracy_list.append(accuracy * label.size()[0])

        loss = sum(loss_list) / access_data.__len__()
        accuracy = sum(accuracy_list) / access_data.__len__()

        # loss = np.around(loss[0].numpy(), decimals=3)
        if mode == "validate":
            print("Epoch ", epoch+1, "/", self.epochs, ". Validation Loss: ",
                  loss, " Validation Accuracy: ", np.around(accuracy, decimals=3))

        if mode == "test":
            print("Testing Data with Best Model. Loss: ", loss, " Accuracy: ",
                  np.around(accuracy, decimals=3))

        return accuracy


def get_accuracy(prediction, actual_value, dict):
    """
    Calculate the accuracy of the model after every batch.

    :param prediction: list of predicted values
    :param actual_value: list of actual values
    :param dict: vocabulary
    :returns: accuracy for this batch
    """
    count = 0

    for i in range(len(prediction)):
        # check if the prediction is correct and not unknown
        if(prediction[i] == actual_value[i] and prediction[i] != len(dict)-1):
            count += 1

    return count/len(prediction)


class Datasets(Dataset):

    def __init__(self, seq_ins, seq_outs):
        """
        Initial function used to get
        embedded training input, output, dictionary

        :param training_input: embedded input
        :param training_output: embedded output
        :param dict: embedded dictionary
        """
        super(Datasets, self).__init__()

        self.seq_ins = seq_ins
        self.seq_outs = seq_outs

    def __getitem__(self, index):
        """
        __getitem__ is a required function of Pytorch if we want to use
        neural network (torch.nn), get the content and corresponding
        label of each word is the index of next word in dictionary.

        :param index: index of word in training set or test set
        :return: content and label of this word
        """

        seq_in = self.seq_ins[index]
        # At the moment, we can only output 1 token, because the size will
        # grow exponentially with the length of the output sequences
        seq_out = self.seq_outs[index][0]

        return seq_in, seq_out

    def __len__(self):
        """
        __len__ is a required function of neural network of Pytorch.
        :return: the length of training set or test set
        """

        return len(self.seq_outs)


class RNN(nn.Module):

    def __init__(self, training_data, dict_size):
        """
        Initial function for RNN.

        :param training_data: embedded training input
        :param dict: embedded dictionary
        """
        super(RNN, self).__init__()

        self.training_data = training_data

        # RNN with 1 input layer, 1 hidden layer, 1 output layer
        # Input layer: 8 unit, hidden layer: 50 unit, output layer: 9 unit
        # The number of unit of output layer = input layer + 1
        # (1 for a unknown word)
        self.RNN = nn.RNN(input_size=dict_size, hidden_size=50,
                          num_layers=1, bidirectional=False)

        # Fully connected layer
        self.fc = nn.Linear(in_features=50, out_features=dict_size)

    def forward(self, input):
        """
        Pipeline for Neural network in Pytorch (build-in function).

        :param input: 2-dimensional tensor ( batch_size x input_size)
        :returns: final output of neural network,
        the dimension of neural network = number of classes
        """

        # Increasing dimension of input by 1
        # Input shape: [batch_size x input_size]
        # Output shape: [1 x batch_size x input_size]

        output, _ = self.RNN(input.float())

        output = output.permute(1, 0, 2)
        output = self.fc(output[-1])

        # print(output)

        return output
