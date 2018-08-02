#!/usr/bin/python3
# This file contains the Class Backpro and some other functions
# please run main.py
import math
import random



class Backpro:  # This is one node, there are total three nodes
    def __init__(self, inpn, hidn, outn, desire, x_in, learning, file):
        ##TODO the number of input node, hidden node and output node.
        self._inpn = inpn  # number of input node
        self._hidn = hidn  # number of hiddwn node
        self._outn = outn  # number of output node
        self._xin = x_in  # xinput
        self._file = file
        self._error = 0  # mean square error
        self._learning = learning  # learning rate
        self._desire = desire  # desire output
        self._momentum, self._epoch = 0, 0
        self._previous_learning, self._previous_momentum = 0, 0
        self._delta_hiddenw = initial_weight(self._hidn, self._inpn + 1)  # hidden layer delta weight
        self._delta_outputw = initial_weight(self._outn, self._hidn + 1)  # output later delta weight
        # initial weight matrix        #self._y is the output from output node, self._yhidden ..from hidden node
        self._hidweight = initial_weight(self._hidn, self._inpn + 1)  # +1 means bias
        self._outweight = initial_weight(self._outn, self._hidn + 1)
        random_weight(self._hidweight)  # (default -0.5,+0.5)
        random_weight(self._outweight)  # uniform from -1 to +1
        print("initial hidden weight: ", self._hidweight, '\n')
        print("initial hidden weight: ", self._hidweight, '\n', file=self._file)
        print("initial output weight: ", self._outweight, '\n')
        print("initial output weight: ", self._outweight, '\n', file=self._file)

    def run(self, epsilon, epoch_criteria, error_criteria, momentum):
        try: 
            Error_log = open("Error_Log.txt", 'w')
            epoch = 1
            terminate = False
            self._momentum = momentum
            while not terminate:
                self.train(epsilon, self._momentum)
                print('error: ', self.get_error())
                print('error: ', self.get_error(), file=Error_log)
                if epoch >= epoch_criteria or self.get_error() <= error_criteria:  # stop criteria
                    hidden_weights = self.get_hiddenw()
                    output_weights = self.get_outw()
                    terminate = True  # stop training
                    print('final training epoch is: {}\n'.format(epoch))
                    print('final training epoch is: {}\n'.format(epoch), file=self._file)
                    print('final training Sum error is: {}\n'.format(self.get_error()))
                    print('final training Sum error is: {}\n'.format(self.get_error()), file=self._file)
                    print('final training Mean square error is: {}\n'.format((1 / len(self._xin)) * self.get_error()))
                    print('final training Mean square error is: {}\n'.format((1 / len(self._xin)) * self.get_error()),
                        file=self._file)
                    print("final hidden_weight is ", hidden_weights, '\n')
                    print("final hidden_weight is ", hidden_weights, '\n', file=self._file)
                    print("final output_weight is ", output_weights, '\n')
                    print("final output_weight is ", output_weights, '\n', file=self._file)
                else:
                    epoch += 1
            self._epoch = epoch
            return (hidden_weights, output_weights)
        except IOError:
            print("file error")
        finally:
            Error_log.close()
    def train(self, epsilon, momentum):
        e = list()  # e is the error
        phi = []  # contain all the phi
        error, yhidden, yout = 0, [], []
        for j, x_i in enumerate(self._xin):  # each pattern,j represents the index of pattern
            # can't use index because some data might have the same value
            x_i.extend([1])  # x0=1,  add bias =1 to the beginning of list x_com
            yhidden = self.hidden(x_i)  # hidden layer function
            yout = self.output(yhidden)  # output layer function
            e = self.cal_error(self._desire[j], yout, epsilon)  # calculate the error
            (self._outweight, phi) = self.modify_outputw(self._outweight, yhidden, yout, self._learning, e,
                                                         momentum)
            self.modify_hiddenw(self._hidweight, x_i, yhidden, self._learning, phi, momentum)
            x_i.pop(-1)  # delete the x0 from x_i
            error += sum(i * j for i, j in zip(e, e))
        self._error = error
        if self._error < 10:
            self._learning = 0.01
            self._momentum = 0.99
        elif self._error < 20:
            self._learning = 0.01
            self._momentum = 0.9
        elif self._error < 50:
            self._learning = 0.05
            self._momentum = 0.85
        elif self._error < 70:
            self._learning = 0.1
            self._momentum = 0.6
        if self._previous_learning != self._learning or self._previous_momentum != self._momentum:
            print("current learning rate is:{}, momemtum is to:{}".format(self._learning, self._momentum))
            self._previous_learning = self._learning
            self._previous_momentum = self._momentum

    def hidden(self, x):
        ahid = []
        for ij in range(self._hidn):
            ahid.append(tanh(sum(i * j for i, j in zip(self._hidweight[ij], x))))  # calculate the dot product
        ahid.extend([1])  # add the bias
        return ahid

    def output(self, x):
        aout = []
        for ij in range(self._outn):
            aout.append(sigmoid(sum(i * j for i, j in zip(self._outweight[ij], x))))  # calculate the dot product
        return aout

    def cal_error(self, desire, true, epsilon):
        e = list()
        for i in range(len(desire)):
            if true[i] >= 1 - epsilon and desire[i] == 1 - epsilon:
                e.append(0)
            elif true[i] <= epsilon and desire[i] == epsilon:
                e.append(0)
            else:
                e.append(desire[i] - true[i])  # it's not the abs
        return e

    def modify_hiddenw(self, weight, h_in, h_out, ln, phi, m):  # m is momentum
        ##TODO modify hidden layer weight
        for ij in range(len(weight)):  # for every node in the hidden layer, do it respectively, reduce bias!!!
            sum_phi = 0  # skip the bias because doesn't have weight
            for j in range(len(phi)):  # I don't update the bias's weight
                sum_phi += phi[j] * self._outweight[j][ij]  # sum(W21*phi_out)
            phi_hid = sum_phi * dtanh(h_out[ij])
            weight[ij] = [i + ln * phi_hid * j + m * w for i, j, w in zip(weight[ij], h_in, self._delta_hiddenw[ij])]
            self._delta_hiddenw[ij] = [ln * phi_hid * j + m * i for i, j in zip(self._delta_hiddenw[ij], h_in)]

    def modify_outputw(self, weight, o_in, o_out, ln, e, m):  # m is momentum
        # the output node weight, the hidden layer output+bias,output node output,learning rate, error
        ##TODO modify output layer weight
        p = list()
        for ij in range(len(weight)):  # there is 3 nodes, deal with them respectively
            phi = e[ij] * dsigmoid(o_out[ij])
            p.append(phi)
            weight[ij] = [i + ln * phi * j + m * w for i, j, w in zip(weight[ij], o_in, self._delta_outputw[ij])]
            self._delta_outputw[ij] = [ln * phi * j + m * i for i, j in zip(self._delta_outputw[ij], o_in)]
        return (weight, p)

    def test(self, xin, epsilon):
        desire = [[1 - epsilon, epsilon, epsilon], [epsilon, 1 - epsilon, epsilon], [epsilon, epsilon, 1 - epsilon]]
        count = list()
        for x_in in xin:
            small = float('inf')  # A big figure
            index = 0
            x_in.append(1)
            yhidden = self.hidden(x_in)
            yout = self.output(yhidden)
            for ij, d in enumerate(desire):
                temp = math.sqrt(sum((i - j) ** 2 for i, j in zip(yout, d)))
                if temp < small:
                    small = temp
                    index = ij
            count.append(index)
            x_in.pop(-1)
        return count

    def get_error(self):
        return self._error

    def get_hiddenw(self):
        return self._hidweight

    def get_outw(self):
        return self._outweight

    def get_epoch(self):
        return self._epoch


def sigmoid(value):
    ##TODO Sigmoid function
    value = 1 / (1 + math.exp(-1 * value))
    return value


def dsigmoid(y):
    ##TODO derivative Sigmoid
    z = y - y ** 2
    return z


def tanh(value):
    ##TODO tanh function
    return math.tanh(value)


def dtanh(value):
    return 1 - value ** 2


def initial_weight(I, J, fill=0.0):
    ##TODO create a two dimensional list at size of I, and each I contains J elements.
    m = list()
    for i in range(I):
        m.append([fill] * J)
    return m


def random_weight(matrix, a=-0.5, b=0.5):
    ij = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            random.seed(ij)
            matrix[i][j] = random.uniform(a, b)  # uniform the initial weight
            ij += 1
