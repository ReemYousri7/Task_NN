#!/usr/bin/env python
# coding: utf-8

# In[20]:


import random

def tanh(x):
    exp_pos = 2.718281828 ** x  
    exp_neg = 2.718281828 ** -x  
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)  

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

def random_weight():
    return random.uniform(-0.5, 0.5)

x1, x2 = 0.1, 0.2
target_o1, target_o2 = 0.1, 0.99

w1, w2 = random_weight(), random_weight()
w3, w4 = random_weight(), random_weight()

b1, b2 = random_weight(), random_weight()

h1_input = (w1 * x1) + (w3 * x2) + b1
h2_input = (w2 * x1) + (w4 * x2) + b1

h1_output = tanh(h1_input)
h2_output = tanh(h2_input)

o1_input = (w1 * h1_output) + (w2 * h2_output) + b2
o2_input = (w3 * h1_output) + (w4 * h2_output) + b2

o1_output = tanh(o1_input)
o2_output = tanh(o2_input)

error_o1 = 0.5 * (target_o1 - o1_output) ** 2
error_o2 = 0.5 * (target_o2 - o2_output) ** 2
total_error = error_o1 + error_o2

learning_rate = 0.1

delta_o1 = (o1_output - target_o1) * tanh_derivative(o1_input)
delta_o2 = (o2_output - target_o2) * tanh_derivative(o2_input)

delta_h1 = (delta_o1 * w1 + delta_o2 * w3) * tanh_derivative(h1_input)
delta_h2 = (delta_o1 * w2 + delta_o2 * w4) * tanh_derivative(h2_input)

w1 -= learning_rate * delta_h1 * x1
w2 -= learning_rate * delta_h1 * x2
w3 -= learning_rate * delta_h2 * x1
w4 -= learning_rate * delta_h2 * x2

b1 -= learning_rate * (delta_h1 + delta_h2)


print("Output 1:", o1_output)
print("Output 2:", o2_output)

print("Error 1:", error_o1)
print("Error 2:", error_o2)
print("Total Error:", total_error)

print("Updated Weights:")
print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}")
print(f"b1: {b1}, b2: {b2}")
print(f"Total Error: {total_error}")


# In[ ]:





# In[ ]:




