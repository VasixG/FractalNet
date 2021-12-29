from utils2d import SampleLayer, Neuron, SampleNeuron, Network, SummingRegressionInputNeuron
import numpy as np
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

input_2d = rgb2gray( np.array(Image.open('sierpinski_triangle_pics\it09\sierpinski_triangle_it9_res500_4096_4096.jpg')))

'''
d = max(math.ceil(math.log(len(input_2d),4)), math.ceil(math.log(len(input_2d),4)))

row_lack = 4**d - len(input_2d)
col_lack = 4**d - len(input_2d[0])

arr  = [[254.9745 for _ in range(4**d)] for _ in range(4**d)]

for i in range(row_lack//2, row_lack//2 + len(input_2d)):
    for j in range(col_lack//2, col_lack//2 + len(input_2d[0])):
        arr[i][j] = input_2d[i - row_lack//2][j - col_lack//2]
np_arr = np.array(arr)'''

size = len(input_2d)

sample_input = []

for i in range(0, size, 2):
    for j in range(0 ,size, 2):
        sample_input.append(input_2d[i][j])
        sample_input.append(input_2d[i][j+1])
        sample_input.append(input_2d[i+1][j])
        sample_input.append(input_2d[i+1][j+1])

I = []

print('hey')

for i in range(len(sample_input)):
    I.append(Neuron(lambda x: 0 if x > 0 else 1))

input_layer = SampleLayer(None, I, 1)

p = int(math.log2(len(sample_input))/2)
k = 3
l = len(sample_input)/16
L = []
L.append(SampleLayer(None, I, 1))
L.append(SampleLayer(L[0], [SampleNeuron(j) for j in range(1, 2**(2*p-1)+1)], 2))

for _ in range(1, p-1):
    n = []
    for j in range(1, int(k * l)+ 1):
        n.append(SampleNeuron(j))
    l /= 4
    k += 1
    L.append(SampleLayer(L[k-3], n, k - 1))

L.append(SampleLayer(L[k-2], [SummingRegressionInputNeuron(j) for j in range(1, int(k * l ))], k - 1))


network = Network(L)

sample_outputs = network.outputs(sample_input)
print('hey x2')
#print(f'sample outputs: {sample_outputs}')

sample = k-1

up_ln = [math.log(2**n) for n in range(1, sample+1)]
#print([math.exp(l) for l in sample_outputs])
#print(f'up ln: {up_ln}')
up_sumln = 1/sample * sum(up_ln)

#print(f'up sumln: {up_sumln}')

inh_up = [ln - up_sumln for ln in up_ln]

#print(f'inh up: {inh_up}')

up_sum = sum(inh**2 for inh in inh_up)

#print(f'up sum: {up_sum}')

down_suminp = 1/sample * sum(sample_outputs)

#print(f'down suminp: {down_suminp}')

inh_down = [output - down_suminp for output in sample_outputs]

#print(f'inh down: {inh_down}')

down_sum = sum(inh_down[sample - 1 - i]*inh_up[i] for i in range(sample))

#print(f'down sum: {down_sum}')

dimension = down_sum/up_sum

print(f'fractal dimension: {dimension}')
