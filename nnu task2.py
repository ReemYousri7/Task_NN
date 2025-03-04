#!/usr/bin/env python
# coding: utf-8

# In[4]:


import random

def tanh(x):
    return (2 / (1 + (2.71828 ** (-2 * x)))) - 1

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

def random_weight():
    return random.uniform(-0.5, 0.5)

x1, x2 = 0.05, 0.1
target_o1, target_o2 = 0.01, 0.99

w1, w2 = random_weight(), random_weight()
w3, w4 = random_weight(), random_weight()

w5, w6 = random_weight(), random_weight()
w7, w8 = random_weight(), random_weight()

b1, b2, b3 = random_weight(), random_weight(), random_weight()

learning_rate = 0.1
iterations = 10000  

for i in range(iterations):
    net_h1 = (w1 * x1) + (w3 * x2) + b1
    net_h2 = (w2 * x1) + (w4 * x2) + b1

    out_h1 = tanh(net_h1)
    out_h2 = tanh(net_h2)

    net_o1 = (w5 * out_h1) + (w6 * out_h2) + b2
    net_o2 = (w7 * out_h1) + (w8 * out_h2) + b3

    o1_output = tanh(net_o1)
    o2_output = tanh(net_o2)

    error_o1 = 0.5 * (target_o1 - o1_output) ** 2
    error_o2 = 0.5 * (target_o2 - o2_output) ** 2
    total_error = error_o1 + error_o2

    delta_o1 = (o1_output - target_o1) * tanh_derivative(net_o1)
    delta_o2 = (o2_output - target_o2) * tanh_derivative(net_o2)

    delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * tanh_derivative(net_h1)
    delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * tanh_derivative(net_h2)

    w5 -= learning_rate * delta_o1 * out_h1
    w6 -= learning_rate * delta_o1 * out_h2
    w7 -= learning_rate * delta_o2 * out_h1
    w8 -= learning_rate * delta_o2 * out_h2

    b2 -= learning_rate * delta_o1
    b3 -= learning_rate * delta_o2

    w1 -= learning_rate * delta_h1 * x1
    w2 -= learning_rate * delta_h1 * x2
    w3 -= learning_rate * delta_h2 * x1
    w4 -= learning_rate * delta_h2 * x2

    b1 -= learning_rate * (delta_h1 + delta_h2)

    if i == 0 or i == iterations - 1:
        print(f"Iteration {i+1} - Total Error: {total_error:.10f}")

print("\nFinal Outputs:")
print(f"o1_output: {o1_output:.10f} ")
print(f"o2_output: {o2_output:.10f} ")


# In[ ]:





# In[ ]:




