from __future__ import annotations
from typing import List, Callable, Optional
import numpy as np
import math
from functools import reduce


class Neuron:
    def __init__(self, activation_function: Callable[[float]]) -> None:
        self.activation_function: Callable[[float], float] = activation_function


    def output(self, inputs: List[float]) -> float:
        return self.activation_function(inputs)


class SampleNeuron(Neuron):
    def __init__(self,n: int) -> None:
        Neuron.__init__(self, activation_function = sampling_function)
        self.n = n


    def output(self,inputs: List[int], layer_number: int) -> int:
        return self.activation_function(self.n, inputs, layer_number)


class SummingRegressionInputNeuron(Neuron):
    def __init__(self,n: int) -> None:
        Neuron.__init__(self, activation_function = summing_input_regression_activation_function)
        self.n = n


    def output(self,inputs: List[int], sample: int) -> int:
        return self.activation_function(self.n, inputs, sample)


class SampleLayer:
    def __init__(self, previous_layer: Optional[SampleLayer], neurons: List[Neuron], number: int) -> None:
        self.previous_layer: Optional[SampleLayer] = previous_layer
        self.neurons: List[Neuron] = neurons
        self.number = number
 

    def outputs(self, inputs: List[float]) -> List[float]:
        if self.previous_layer is None:
            self.output = [self.neurons[0].output(inputs[i]) for i in range(len(inputs))]
        else:
            self.output = [n.output(inputs, self.number) for n in self.neurons]
        return self.output


class Network:
    def __init__(self, layers: List[SampleLayer]) -> None:
        self.layers = layers
    

    def outputs(self, input: List[float]) -> List[float]:
        return reduce(lambda inputs, layer: layer.outputs(inputs), self.layers, input)

#calculate value for n neuron in d layer
def devisor_step(n: int, p: int):
    if n%p == 0:
        return 0
    else:
        return 1

#activation function for data samping
def sampling_function(n: int, inputs:  List[float], number_layer: int) -> int:
    real_inputs = []
    if number_layer == 2:
        real_inputs.append(inputs[4*((n-1)//2)])
        real_inputs.append(inputs[1 + 4*((n-1)//2)])
        real_inputs.append(inputs[2 + 4*((n-1)//2)])
        real_inputs.append(inputs[3 + 4*((n-1)//2)])
    else:
        if devisor_step(n, number_layer) == 0:
            real_inputs.append(inputs[(n-1)%(number_layer) + ((n-2)//(number_layer))*4*(number_layer - 1) - 1])
            real_inputs.append(inputs[(n-1)%(number_layer) + number_layer - 1 + ((n-2)//(number_layer))*4*(number_layer - 1) - 1])
            real_inputs.append(inputs[(n-1)%(number_layer)+ 2*(number_layer - 1) + ((n-2)//(number_layer))*4*(number_layer - 1) - 1])
            real_inputs.append(inputs[(n-1)%(number_layer) + 3*(number_layer - 1) + ((n-2)//(number_layer))*4*(number_layer - 1) - 1])
        else:
            real_inputs.append(inputs[(n)%(number_layer) + ((n-1)//(number_layer))*4*(number_layer - 1) - 1])
            real_inputs.append(inputs[(n)%(number_layer) + number_layer - 1 + ((n-1)//(number_layer))*4*(number_layer - 1) - 1])
            real_inputs.append(inputs[(n)%(number_layer)+ 2*(number_layer - 1) + ((n-1)//(number_layer))*4*(number_layer - 1) - 1])
            real_inputs.append(inputs[(n)%(number_layer) + 3*(number_layer - 1) + ((n-1)//(number_layer))*4*(number_layer - 1) - 1])


    return devisor_step(n, number_layer)*np.sum(real_inputs) + (1 - devisor_step(n, number_layer))*np.max(real_inputs)

#summing ln function
def summing_input_regression_activation_function(n: int, inputs: List[float], sample: int):
    real_inputs = []
    real_inputs.append(inputs[n-1])
    real_inputs.append(inputs[n-1+sample])
    real_inputs.append(inputs[n-1+2*sample])
    real_inputs.append(inputs[n-1+3*sample])

    if sum(real_inputs) == 0:
        return 0
    return math.log(sum(real_inputs))

