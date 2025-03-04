#!/usr/bin/env python
# coding: utf-8

# In[10]:


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

w5, w6 = random_weight(), random_weight()
w7, w8 = random_weight(), random_weight()

b1, b2, b3 = random_weight(), random_weight(), random_weight()

print("\n** Initial Weights and Biases **")
print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}")
print(f"w5: {w5}, w6: {w6}, w7: {w7}, w8: {w8}")
print(f"b1: {b1}, b2: {b2}, b3: {b3}")

# ** Forward Propagation **
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

print("\n** Forward Propagation **")
print(f"net_h1: {net_h1}, net_h2: {net_h2}")
print(f"out_h1: {out_h1}, out_h2: {out_h2}")
print(f"net_o1: {net_o1}, net_o2: {net_o2}")
print(f"o1_output: {o1_output}, o2_output: {o2_output}")
print(f"Error_o1: {error_o1}, Error_o2: {error_o2}")
print(f"Total Error: {total_error}")

# ** Backpropagation **
learning_rate = 0.1

delta_o1 = (o1_output - target_o1) * tanh_derivative(net_o1)
delta_o2 = (o2_output - target_o2) * tanh_derivative(net_o2)

delta_h1 = (delta_o1 * w5 + delta_o2 * w7) * tanh_derivative(net_h1)
delta_h2 = (delta_o1 * w6 + delta_o2 * w8) * tanh_derivative(net_h2)

print("\n** Backpropagation: Deltas **")
print(f"delta_o1: {delta_o1}, delta_o2: {delta_o2}")
print(f"delta_h1: {delta_h1}, delta_h2: {delta_h2}")

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

print("\n** Updated Weights and Biases **")
print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}")
print(f"w5: {w5}, w6: {w6}, w7: {w7}, w8: {w8}")
print(f"b1: {b1}, b2: {b2}, b3: {b3}")
print(f"Total Error After Update: {total_error}")


# In[ ]:





# In[ ]:




