# VR Assignment 3

*Subhajeet Lahiri - IMT2021022*
*Sai Madhavan G - IMT2021101*
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
