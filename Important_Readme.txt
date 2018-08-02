Created on 10-FEB-2018
@author: Huicheng Liu
 This is the main code, please run it to execute the code.
					!!Attention!! 
"""<Please make sure to let the code end by itself even though it might take about 5 minutes to finish(depends on the computer). If not, the output file "Final_output_result.txt" which contains all the informations won't show anything. If you think it costs to much time, open the file called "sample_output_result.txt" and use it as a reference. This file contains the exactly same result information which is from the previous successfully run>"""

 
Requirements: Python3, Python3.6.3 is prefered(doesn't work on python 2). My code doesn't include any library, all the import file are wrote by myself which contains class and functions, the only dependent import is "math" and "random" that includes in python3 so there shouldn't be any problem running the code. 
Run the code simply through the python shell, Pycharm, Visual studio(not suggested! It might take a year to finish it!!)or even through the command line on Mac and Linux by typing "python3 main.py". The input file is "assignment2train.csv" and "assignment2test.csv", the output file "Final_output_result.txt" contains all the information including epoch(iteration time), learning rate, momentum value, activation function, initial weight, final weight, classify result, precision, recall, classification accuracy, confusion matrix etc(or you can just read those informations from the console). 
The file "Error_Log.txt" contains some runtime information such as error changes after each epoch, momentum value changes, learning rate changes etc.
After trial and error, the parameters in the network has been set to optimal, please do not change any of the parameters or else the network might work bad. 

The best Result I got is about 92.982% accuracy!
The .txt files format might be wrong if you open it in windows, so I suggest to open the txt file
with a app such as notepad or other editor.

My decision:
1.Initial weight satisfies a uniform distribution which range is (-0.5,0.5), this is referenced from the lecture nodes.

2.My activation function for the hidden node is tanh function, and the output node function is sigmoid function. I tried all kinds of other activation functions such as ReLUs, Leaky ReLUs, ELUs and Softplus. Eventually, after various combinations, I found that using tanh function for the hidden node and sigmoid function for the output node is the best.

3. My learning rate and momentum is adaptive, the initial learning rate and momentum will be set to 0.1 and 0.4. Along with the sum error decreasing, my learning rate will start decreasing and my momentum will start increasing. When the sum error is smaller than 100, the learning rate will decrease to 0.08 and momentum will increase to 0.6. When the sum error drops to 50, the learning rate will be 0.05 and the momentum will be 0.85 and so on. These values are selected by lots of testing so the values are optimal.

4. My iteration criteria is set 500 and the sum error criteria has been set to 50. I found that my neural network will be very hard to train after the sum error drops down to 34. Since it takes less time to train the net work, the mean square error is only 0.024 and I can reach a 92.982% accuracy, I eventually decided to set the stop error criteria to 50. Usually, the code will stop with the final iteration around 330.

5. My network has 1 input layer with 11 input nodes which refers to 11 features, 1 hidden layer with 58 hidden nodes and 1 output layer with 3 output nodes that can represent three different classes. I choose to add 58 hidden layer because it gives me the best result. I tried different numbers of hidden nodes vary from 12 to 60 and finally found that 58 is the best choice!

6. The momentum value decision has been answered together with learning rate decision in No.3. I have some further explanation here, The reason I choose to increase the momentum and decrease the learning rate when the sum error drops to a certain value is to avoid the local minima. My network won't have any progress if the learning rate and momentum remained in a certain value, the sum error will keep floating up and down.

7. I did the Data preprocessing because I found that my neural network won't work if I use the original data. According to the data, some features have really small values such as 0.032 where as some others might even reaches 133. The feature which has the smaller value will almost have no effect to the output and weight update. By normalizing the data using Gaussian Normalization((x-x_mean)/std  x_mean refers to the mean, std refers to the standard deviation) makes the network better.

8. According to Baum and Haussler, I should use 1200 training data since I used more than 60 weights and wants to reach a 95 percent accuracy. Hence, I split the data to three parts, 70% training data, 15% validation data and 15% testing data, each data set contains all three classes to make sure the data sets are valid.

