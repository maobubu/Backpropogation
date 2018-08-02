#!/usr/bin/python3
'''
This file contains the class Data and function arrange_train
arrange_Test, label ,split desire etc
please run main.py
'''
import math


class Data:
    def __init__(self, train, test, output, theta):  # theta means epsilon
        self._train_in = train
        self._test_in = test
        self._out = output
        self._x_new = []
        self._desire_new = []
        self._test_new = []
        self._test_desire = []
        self._final = []
        self._theta = theta

    def arrange_train(self):
        ##TODO transfer each line of the txt file into a list including x and label
        for line in self._train_in:
            train_out = line.rstrip().split(',')
            self._desire_new.append(self.label(train_out))  # add the desire output to a list
            train_out.pop(-1)  # delete the labels # and appended into a list
            train_out = list(map(float, train_out))
            self._x_new.append(train_out)  # append each list to a new list.
        self._x_new = self.preprocess(self._x_new)

    def arrange_test(self):
        for line in self._test_in:
            test_out = line.rstrip().split(',')
            self._final.append(line)
            self._test_desire.append(self.label(test_out))  # add the desire output to a list
            test_out.pop()  # delete the labels # and appended into a list
            test_out = list(map(float, test_out))
            self._test_new.append(test_out)  # append each list to a new list.
        self._test_new = self.preprocess(self._test_new)  # normalize the testing data!!

    def preprocess(self, x):  # gaussian normalization
        for j in range(len(x[0])):  # 0-10
            mean, std = 0, 0
            for i in range(len(x)):  # 0-2xxx
                mean += x[i][j] / len(x)
            for i in range(len(x)):
                std += ((x[i][j] - mean) ** 2) / len(x)
            std = math.sqrt(std)
            for i in range(len(x)):
                x[i][j] = (x[i][j] - mean) / std
        return x

    def label(self, label_list):
        ##TODO transform the strings to vectors.
        if label_list[11] == '5':
            label_list[11] = [1 - self._theta, self._theta, self._theta]

        elif label_list[11] == '7':
            label_list[11] = [self._theta, 1 - self._theta, self._theta]

        elif label_list[11] == '8':
            label_list[11] = [self._theta, self._theta, 1 - self._theta]
        return label_list[11]

    def labelprevious(self, label_list):
        ##TODO transform the strings to vectors for the assignment one data
        if label_list[4] == 'Iris-setosa':
            label_list[4] = [1 - self._theta, self._theta, self._theta]
        elif label_list[4] == 'Iris-versicolor':
            label_list[4] = [self._theta, 1 - self._theta, self._theta]
        elif label_list[4] == 'Iris-virginica':
            label_list[4] = [self._theta, self._theta, 1 - self._theta]
        return label_list[4]

    def split_desire(self, desire):
        # TODO split the desire output
        d1 = [float(l[0]) for l in desire]
        d2 = [float(l[1]) for l in desire]
        d3 = [float(l[2]) for l in desire]
        dic = dict(first=d1, second=d2, third=d3)  # add to the dictionary
        return dic

    def get_x(self):
        return self._x_new

    def get_desire(self):
        return self._desire_new

    def get_test_desire(self):
        return self._test_desire

    def get_test_new(self):
        return self._test_new

    def get_final_class(self):
        return self._final
