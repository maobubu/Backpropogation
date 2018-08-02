#!/usr/bin/python3
'''
This file contains the final function
please run main.py
'''
import random
import math


def final(final, desire, true, sum_error, output_file, d_final):
    c1 = [i for i, e in enumerate(true) if e == 0]  # the index of true output 0
    c2 = [i for i, e in enumerate(true) if e == 1]  # the index of true output 1
    c3 = [i for i, e in enumerate(true) if e == 2]  # the index of true output 2
    m1 = [i for i, e in enumerate(desire) if e == 0]  # the index of desire output 0
    m2 = [i for i, e in enumerate(desire) if e == 1]  # the index of desire output 1
    m3 = [i for i, e in enumerate(desire) if e == 2]  # the index of desire output 2
    miss_match = [i for (i, e), j in zip(enumerate(true), desire) if e != j]  # find the miss match for overall
    p1 = len(set(m1) & set(c1))  # find the miss match for class one
    p2 = len(set(m2) & set(c2))  # find the miss match for class two
    p3 = len(set(m3) & set(c3))  # find the miss match for class three
    accuracy = (1 - len(miss_match) / len(desire)) * 100  # classification error in percent
    pre1, pre2, pre3 = p1 / len(c1), p2 / len(c2), p3 / len(c3)
    re1, re2, re3 = p1 / len(m1), p2 / len(m2), p3 / len(m3)
    print(" final testing output:\n(0.9,0.1,0.1) represent '5'\n(0.1,0.9,0.1)represent '7'\n"
          "(0.1,0.1,0.9) represents '8'\n")
    print(" final testing output:\n(0.9,0.1,0.1) represent '5'\n(0.1,0.9,0.1)represent '7'\n"
          "(0.1,0.1,0.9) represents '8'\n", file=output_file)
    ##TODO for Quality 5
    print("\nClassified as Quality 5:")
    print("\nClassified as Quality 5:", file=output_file)
    if not c1:  # execute when c1 is empty
        print('None of them have been classified to Quality 5\n')
        print('None of them have been classified to Quality 5\n', file=output_file)
    for i in c1:
        print(final[i])
        output_file.write(final[i])
        output_file.write(final[i])
    ##TODO for Quality 7
    print("\nClassified as Quality 7:")
    print("\nClassified as Quality 7:", file=output_file)
    if not c2:
        print('None of them have been classified to Quality 7\n')
        print('None of them have been classified to Quality 7\n', file=output_file)
    for i in c2:
        print(final[i])
        output_file.write(final[i])
    ##TODO for Quality 8:
    print("\nClassified as Quality 8:")
    print("\nClassified as Quality 8:", file=output_file)
    if not c3:
        print('None of them have been classified to Quality 8\n')
        print('None of them have been classified to Quality 8\n', file=output_file)
    for i in c3:
        print(final[i])
        output_file.write(final[i])
        ##TODO add precision and recall,error,sum square error
    print('\nTraining Sum error is {}\n'.format(sum_error))
    print('\nTraining Mean squared error is {}\n'.format((1 / len(d_final)) * sum_error))
    print('Training Sum error is {}\n'.format(sum_error), file=output_file)
    print('Training Mean squared error is {}\n'.format((1 / len(d_final)) * sum_error), file=output_file)
    print('\n\nThe precision and recall\n')
    print('\n\nThe precision and recall\n', file=output_file)

    if not c1:  # execute when c1 is empty
        print('None of them have been classified to Quality 5\n')
        print('None of them have been classified to Quality 5\n', file=output_file)
    else:
        print('Class one Quality 5:\nPrecision:{}\nRecall:{}\n'.format(pre1, re1))
        print('Class one Quality 5:\nPrecision:{}\nRecall:{}\n'.format(pre1, re1), file=output_file)
    if not c2:
        print('None of them have been classified to Quality 7\n')
        print('None of them have been classified to Quality 7\n', file=output_file)
    else:
        print('Class two Quality 7:\nPrecision:{}\nRecall:{}\n'.format(pre2, re2))
        print('Class two Quality 7:\nPrecision:{}\nRecall:{}\n'.format(pre2, re2), file=output_file)
    if not c3:
        print('None of them have been classified to Quality 8\n')
        print('None of them have been classified to Quality 8\n', file=output_file)
    else:
        print('Class three Quality 8:\nPrecision:{}\nRecall:{}\n'.format(pre3, re3))
        print('Class three Quality 8:\nPrecision:{}\nRecall:{}\n'.format(pre3, re3), file=output_file)
    print('The Classification Accuracy is {}%\n'.format(accuracy))
    print('The Classification Accuracy is {}%\n'.format(accuracy), file=output_file)
    print('Draw the confusion matrix；\n')
    print('Draw the confusion matrix；\n', file=output_file)
    confusion_matrix(c1, c2, c3, m1, m2, m3, p1, p2, p3, output_file)


def test(xin, epsilon):
    ##TODO transfer tehe desire outpuy
    desire = [[1 - epsilon, epsilon, epsilon], [epsilon, 1 - epsilon, epsilon], [epsilon, epsilon, 1 - epsilon]]
    count = list()
    for x_in in xin:
        if x_in == desire[0]:
            count.append(0)
        elif x_in == desire[1]:
            count.append(1)
        elif x_in == desire[2]:
            count.append(2)
    return count


def confusion_matrix(c1, c2, c3, m1, m2, m3, p11, p22, p33, output_file):
    p12 = len(set(c1) & set(m2))
    p13 = len(set(c1) & set(m3))
    p21 = len(set(c2) & set(m1))
    p23 = len(set(c2) & set(m3))
    p31 = len(set(c3) & set(m1))
    p32 = len(set(c3) & set(m2))
    print('''----------------------------------Actual Class-----------------------
                                
                                 Quality '5'  |    Quality '7'   |   Quality '8'  
                                              |                  |
                |   Quality "5"       {}      |        {}        |      {}
                |   ---------------------------------------------------------------              
Predicted Class |   Quality "7"       {}      |        {}        |      {}
                |   ---------------------------------------------------------------             
                |   Quality "8"       {}      |        {}        |      {}
    
      
    '''.format(p11, p12, p13, p21, p22, p23, p31, p32, p33))
    print('''----------------------------------Actual Class-----------------------

                                     Quality '5'  |    Quality '7'   |   Quality '8'  
                                                  |                  |
                    |   Quality "5"       {}      |        {}        |      {}
                    |   ---------------------------------------------------------------              
    Predicted Class |   Quality "7"       {}      |        {}        |      {}
                    |   ---------------------------------------------------------------             
                    |   Quality "8"       {}      |        {}        |      {}


        '''.format(p11, p12, p13, p21, p22, p23, p31, p32, p33), file=output_file)
