# VR Assignment 3

*Subhajeet Lahiri - IMT2021022* |
*Sai Madhavan G - IMT2021101* |
*Aditya Garg - IMT2021545*

# Part A

## Problem Statement

The goal of this part of the assignment was to play around with CNNs. Particularly, we were to experiment with different optimization techniques and activation functions. We were to compare the different approaches using training time and classification performance. Based on our experiments, we were to recommend the best architecture.

## Dataset

We performed these experiments on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image classification dataset. It consisted of 50000 train images and 10000 test images of size 32x32x3. Each image had exactly one class label among 10 possible values.

## Base model

We opted to work with a CNN with 3 convolutional layers followed by 2 fully connected layers. This model consisted of around 820k parameters. The model was implemented using the pytorch library. All the code used in this section can be accessed in the `part_a` directory of our [github repo](https://github.com/Gapes21/Assignment-3).

## Evaluation Metrics

We used the following evaluation metrics to compare the results of our experiments:

- **Accuracy:** The percentage of the test set predictions that matched the ground truth.
- **Macro f1:** The average f1 score of all classes.
- **Epochs until convergence:** The number of epochs the model took before it reached the maximum accuracy.
- **Time until convergence:** The time since the beginning of the training till the model reached the maximum accuracy.

## Experiments on optimization strategies

### Vanilla SGD

We started our experiments by optimizing our model using vanilla stochastic gradient descent (SGD). We dub this experiment and its resulting model as `CNN1`. Our implementation used the `SGD` optimizer of `torch.optim` and we used a constant learning rate of 10$^{-3}$.

#### Observations

- The model came close to convergence after 500 epochs and 7430 seconds (~2 hours).
- It reached the best test loss at around 200 epochs after which the test loss began rising
- Despite the increasing test loss, the test accuracy and f1 kept increasing and the train accuracy reached 99% at the end of 500 epochs.
- The best accuracy was **63.67%** and the best test f1 was **0.64** at **500** epochs (~**2 hours**)

### SGD with momentum
We next tried introducing momentum to our optimization. We used a momentum of **0.9** with initial learning rate at $10^{-3}$. This experiment and it's accompanying model was called `CNN1.1`.

#### Observations
- The least test loss was obtained in just 40 epochs (789 seconds).
- Similar to the previous experiment, despite the increasing test loss, the test accuracy and f1 were increasing up until the experiment concluded at 120 epochs.
- The best accuracy was **70.83%** and the best test f1 was **0.71** at **120** epochs (~**40 minutes**)

### Adam
We then used the Adam optimizer which uses adaptive estimates of first order and second order moments. The initial learning rate was again $10^{-3}$. This experiment and it's accompanying model was called `CNN1.2`.

#### Observations
- The least test loss along with the maximum accuracy was obtained in just 7 epochs (433 seconds).
- Contrary to the earlier experiments, the test accuracy and f1 was fluctuating with a downward trend following this, when the test loss started increasing.
- The best accuracy was **68.17%** and the best test f1 was **0.69** at **7** epochs (~**7 minutes**)

### Conclusion
- The vanilla SGD approach was very slow and it wasn't able to reach the best accuracy score either.
- Using momentum made training considerably faster and the model reached a much better score too.
- Both of the above models exhibited an oddity wherein the test accuracy was increasing in spite of signs of over-fitting, something we weren't able to explain.
- Adam was the fastest to converge. However, it's best score was a bit lesser than the approach of SGD with momentum. This is probably due to the *"overshooting"* nature of the optimizer.
- We choose **Adam** to be our optimizer of choice due to it's fast convergence. We can counteract this overshooting nature in larger models and datasets using weight decay.

## Experiments using various values of dropout

In the earlier experiments, we observed overfitting to be a persistent problem. To fix this, we use different values of dropout in the fully connected layers. We conducted experiments with 4 different values of dropout:
1. `CNN2.1`: **0.2**
2. `CNN2.2`: **0.4**
3. `CNN2.3`: **0.6**
4. `CNN2.4`: **0.8**

We trained these three models for 25 epochs each. All other factors except the dropout of these models are identical to model `CNN1.2`

### Observations

- The best accuracy of `CNN2.1` was **71.44%** and the best test f1 was **0.71** at **9** epochs (**83 seconds**)
- The best accuracy of `CNN2.2` was **72.54%** and the best test f1 was **0.72** at **17** epochs (**150 seconds**)
- The best accuracy of `CNN2.3` was **72.00%** and the best test f1 was **0.72** at **14** epochs (**127 seconds**)
- The best accuracy of `CNN2.4` was **69.59%** and the best test f1 was **0.69** at **22** epochs (**200 seconds**)
### Conclusion
- Using dropout increased performance considerably at the cost of a slight increase in convergence time
- Using a dropout value of **0.4** seemed to be ideal as increasing it too much is resulting in decrease in performance.

## Experiment with addition of Batch Normalization

In this experiment, we introduced 2-dimensional batch normalization after each convolutional layer. We called this model `CNN3` All other factors were identical to the model `CNN2.2`.

### Observations
- The best accuracy of `CNN3` was **75.89%** and the best test f1 was **0.76** at **25** epochs (**227 seconds**)

### Conclusion
- Using batch normalization further increased the score. We speculate this is because it helps in solving the vanishing/exploding gradients problem, thereby making more neurons "active".

## Experiments with various activation functions

Finally, we experimented with different activation functions that were used after each convolutional and fully connected layers. All models upto `CNN3` used the ReLU activation function. All other factors except the activation function were identical to `CNN3`. We conducted experiments with the following activation functions:
1. `CNN4.1`: Sigmoid
2. `CNN4.2`: tanh
3. `CNN4.3`: leaky ReLU (negative_slope = 0.01)
4. `CNN4.4`: GELU

### Observations
- The best accuracy of `CNN4.1` was **72.67%** and the best test f1 was **0.73** at **25** epochs (**230 seconds**)
- The best accuracy of `CNN4.2` was **72.84%** and the best test f1 was **0.73** at **23** epochs (**211 seconds**)
- The best accuracy of `CNN4.3` was **76.77%** and the best test f1 was **0.77** at **11** epochs (**105 seconds**)
- The best accuracy of `CNN4.4` was **77.21%** and the best test f1 was **0.77** at **22** epochs (**203 seconds**)

### Conclusion
- Based on the above observations, we concluded that the **GELU** activation function was the best.

___
## Conclusion
| **Model** | **Optimization Strategy** | **Dropout** | **Batch Normalization** | **Activation Function** | **Test Accuracy (%)** | **Test f1** | **Epochs till convergence** | **Time till convergence (s)** |
| --------- | ------------------------- | ----------- | ----------------------- | ----------------------- | --------------------- | ----------- | --------------------------- | ----------------------------- |
| CNN1      | SGD                       | 0           | No                      | ReLU                    | 63.67                 | 0.64        | 500                         | 7430                          |
| CNN1.1    | SGD with momentum         | 0           | No                      | ReLU                    | 70.83                 | 0.71        | 120                         | 2464                          |
| CNN1.2    | Adam                      | 0           | No                      | ReLU                    | 68.17                 | 0.68        | *7*                         | 433                           |
| CNN2.1    | Adam                      | 0.2         | No                      | ReLU                    | 71.44                 | 0.71        | 9                           | *83*                          |
| CNN2.2    | Adam                      | 0.4         | No                      | ReLU                    | 72.54                 | 0.72        | 17                          | 150                           |
| CNN2.3    | Adam                      | 0.6         | No                      | ReLU                    | 72.00                 | 0.72        | 14                          | 127                           |
| CNN2.4    | Adam                      | 0.8         | No                      | ReLU                    | 69.59                 | 0.69        | 22                          | 200                           |
| CNN3      | Adam                      | 0.4         | Yes                     | ReLU                    | 75.89                 | 0.76        | 25                          | 227                           |
| CNN4.1    | Adam                      | 0.4         | Yes                     | Sigmoid                 | 72.67                 | 0.73        | 25                          | 230                           |
| CNN4.2    | Adam                      | 0.4         | Yes                     | tanh                    | 72.84                 | 0.73        | 23                          | 211                           |
| CNN4.3    | Adam                      | 0.4         | Yes                     | Leaky ReLU              | 76.77                 | *0.77*      | 11                          | 105                           |
| *CNN4.4*  | Adam                      | 0.4         | Yes                     | GELU                    | *77.21*               | *0.77*      | 22                          | 203                           |

Based on the results of our experiments, we recommend the configuration of the model `CNN4.4`, due to it's performance and efficiency.
___
