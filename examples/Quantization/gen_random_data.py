import numpy as np


input_data = np.random.rand(10, 6)

params_data = np.random.rand(10, 6)  

with open('input_data.txt', 'w') as f:
    for row in input_data:
        f.write(' '.join(map(str, row)) + '\n')

with open('params_data.txt', 'w') as f:
    for row in params_data:
        f.write(' '.join(map(str, row)) + '\n')

print("gen done!")
