import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('Advertising.csv')
x = dataframe.values[:, 2]
y = dataframe.values[:, 4]
# print(y)
# plt.scatter(x, y, marker='o')
# plt.show()


def predict(new_radio, weight, bias):

    return weight * new_radio + bias


def cost_function(x, y, weight, bias):
    n = len(x)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight * x[i] * bias))**2
    return sum_error/n


def update_weight(x, y, weight, bias, learning_rate):
    n = len(x)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2*x[i]*(y[i] - (x[i]*weight + bias))
        bias_temp += -2*(y[i] - (x[i]*weight + bias))
    weight -= (weight_temp/n)*learning_rate
    bias -= (bias_temp/n)*learning_rate

    return weight, bias


def train(x, y, weight, bias, learning_rate, iter):
    cost_his = []
    for i in range(iter):
        weight, bias = update_weight(x, y, weight, bias, learning_rate)
        cost = cost_function(x, y, weight, bias)
        cost_his.append(cost)

    return weight, bias, cost_his


weight, bias, cost = train(x, y, 0.03, 0.0014, 0.001, 60)

# print(weight, bias, cost)

print(predict(19, weight, bias))


solanlap = [i for i in range(60)]
plt.plot(solanlap, cost)
plt.show()