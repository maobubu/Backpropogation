#!/usr/bin/python3
'''
Created on 10-FEB-2018
@author: Huicheng Liu
This is the main code, please run it to execute the code.
					                    !!Attention!!
"""<Please make sure to let the code end by itself even though it might take about 5 minutes to finish(depends on the computer).
If not, the output file "Final_output_result.txt" which contains all the informations won't show anything. If you think it costs
too much time, open the file called "sample_output_result.txt" and use it as a reference. This file contains the exactly same result
nformation which is from the previous successfully run>"""
 
Requirements: Python3, Python3.6.3 is prefered(doesn't work on python 2). My code doesn't include any library, all the import file
are wrote by myself which contains class and functions, the only dependent import is "math" and "random" that includes in python3
so there shouldn't be any problem running the code.
Run the code simply through the python shell, Pycharm, Visual studio(not suggested! It might take a year to finish it!!)or even
through the command line on Mac and Linux by typing "python3 main.py". The input file is "assignment2train.csv" and "assignment2test.csv",
the output file "Final_output_result.txt" contains all the information including epoch(iteration time), learning rate, momentum value,
activation function, initial weight, final weight, classify result, precision, recall, classification accuracy, confusion matrix etc.
The file "Error_Log.txt" contains some runtime information such as error changes after each epoch, momentum value changes, learning rate
changes etc(or you can just read these informations from the console).
After trial and error, the parameters in the network has been set to optimal, please do not change any of the parameters
or else the network might work bad.

The best Result I got is about 92.982% accuracy!

---------------------------------------------
'''
print("This is a Backpropogation algorithm\nauthor: Huicheng Liu\nDate:2/10/2018")
# BEGIN : variable declaration
import Data as Dt
import Backpro_maobu as Bp
import Final as Fn

learning_rate = 0.1  # 0.9, 0.8 is the best for assignment 1 data(sigmoid),
# 0.2225, 0.0 is the best for assignment two(sigmoid).
# 0.05,0.8 is the best for tanh, assignment 1
# 0.05, 0.36 is the best for tanh, assigment 2
epoch = 500  # stop criteria iteration time or 1000
bias = 1
input_node, hidden_node, output_node = 11, 58, 3
momentum = 0.4
# momentum set to 0.4 also good for assignment 2(sigmoid)
epsilon = 0.1  # the desire out transform theta
error_criteria = 50  # the error stop criteria
train_file = open('assignment2train.csv', 'r')
test_file = open('assignment2test.csv', 'r')


def main():
    try:
        output_file = open('Final_output_result.txt', 'w')
        print('start:\nlearning rate has been set to:{}\niteration criteria'
            ' has been set to:{}\n'.format(learning_rate, epoch))
        print(
            'error criteria has been set to:{}\nmomentum has been set to:{}\nbias has been set to:{}'.format(error_criteria,
                                                                                                            momentum,
                                                                                                            bias))
        print(
            'start:\nlearning rate has been set to:{}\niteration criteria has been set to:{}'.format(learning_rate, epoch),
            file=output_file)
        print(
            'error criteria has been set to:{}\nmomentum has been set to:{}\nbias has been set to:{}'.format(error_criteria,
                                                                                                            momentum,
                                                                                                            bias),
            file=output_file)
        d = Dt.Data(train_file, test_file, output_file, epsilon)
        ##TODO training phase
        d.arrange_train()
        # d_final = d.split_desire(d.get_desire())  # a dictionary that contains all the desire output
        d_final = d.get_desire()
        d.arrange_test()
        print("1 input layer, 1 hidden layer, 1 output layer\n")
        print("1 input layer, 1 hidden layer, 1 output layer\n", file=output_file)
        print("the number of input node is: {}, hidden node is: {}, output node is: {}\n".format(input_node, hidden_node,
                                                                                                output_node))
        print("the number of input node is: {}, hidden node is: {}, output node is: {}\n".format(input_node, hidden_node,
                                                                                                output_node),
            file=output_file)
        print(" hidden node activation function is tanh function, output node activation function is sigmoid function")
        print(" hidden node activation function is tanh function, output node activation function is sigmoid function",
            file=output_file)

        # set to 4,4,3 is the best for the assignment 1 data
        # set to 11,58,3 is the best for 400 data assignment 2
        print("Start the training phase\n")
        print("Start the training phase\n", file=output_file)
        b1 = Bp.Backpro(input_node, hidden_node, output_node, d_final, d.get_x(),
                        learning_rate, file=output_file)  # first:weights,second:desire output
        b1.run(epsilon, epoch, error_criteria, momentum)
        print("Start the testing phase\n")
        print("Start the testing phase\n", file=output_file)
        true = b1.test(d.get_test_new(), epsilon)
        desire = Fn.test(d.get_test_desire(), epsilon)
        print('final Training epoch is: {}\n'.format(b1.get_epoch()))
        print('final Training epoch is: {}\n'.format(b1.get_epoch()), file=output_file)
        print('final Training Sum error is: {}\n'.format(b1.get_error()))
        print('final Training Sum error is: {}\n'.format(b1.get_error()), file=output_file)
        print('final Training Mean squared error is: {}\n'.format((1 / len(d_final)) * b1.get_error()))
        print('final Training Mean squared error is: {}\n'.format((1 / len(d_final)) * b1.get_error()), file=output_file)
        Fn.final(d.get_final_class(), desire, true, b1.get_error(), output_file, d_final)
        print('final Training epoch is: {}\n'.format(b1.get_epoch()))
        print('final Training epoch is: {}\n'.format(b1.get_epoch()), file=output_file)
    except IOError:
        print('File Error')
    finally:
        output_file.close()

if __name__ == "__main__": main()
