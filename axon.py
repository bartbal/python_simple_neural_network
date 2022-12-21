'''
File: axon.py
Project: Insmart_v2
File Created: Wednesday, 14th December 2022 12:21:31 pm
Author: Bart van Netburg (b.van.netburg@insmart.nl)
-----
Last Modified: Wednesday, 21st December 2022 11:03:51 am
Modified By: Bart van Netburg (b.van.netburg@insmart.nl>)
-----
Copyright 2022 - 2022 Insmart B.V., Insmart
'''

import sys
import numpy as np
from numpy import random
import neuron

class axon:

    # the weight of this axon
    weight : float = None

    # the input neuron
    input : neuron = None

    # the output neuron
    output : neuron = None

    def __init__(self, input : neuron, weight : float = None) -> None:
        if(weight == None):
            weight = random.rand()
        self.input = input
        self.weight = weight

    """
    @brief
    set the output neuron of this axon
    """
    def set_output(self, neuron_out : neuron) -> None:
        if self.output == None:
            self.output = neuron_out
        else:
            print("ERROR: axon output already set!")
            print(neuron_out.printName())

    """
    @brief
    Get the weight
    """
    def get_weight(self) -> float:
        return self.weight

    """
    @brief
    Set the weight
    """
    def set_weight(self, new_weight : float) -> None:
        self.weight = new_weight
    
    """
    @brief
    Get the weigted delta. 
    @details
    weight * output.delta
    """
    def get_weighted_delta(self) -> float:
        return self.weight * self.output.get_delta()

    """
    @brief
    Get the output of this axon
    """
    def get_output(self) -> float:
        return self.input.get_output()*self.weight

    """
    @brief
    Print self.weight
    """
    def print(self):
        print("    "+str(self.weight))

    """
    @brief
    call the chain_print of the input neuron
    """
    def chain_print(self):
        self.input.chain_print()

    """
    @brief 
    print the last output of the input neuron and its weights
    """
    def input_print(self):
        self.input.print()
        self.input.weights_print()

    """
    @brief
    learn the input of this axon
    """
    def learn(self, zeta : float = 0.1, expected_out : float = None):
        self.input.learn(zeta, expected_out)


    """
    @brief 
    calculate new weight for this axon
    @param float zeta
    the amount to increment the weight with
    """
    def update_weight(self, zeta : float, delta : float) -> None:
        #weight zeta * delta * self.input.get_last_output()
        self.weight = self.weight + zeta * delta * self.input.get_last_output()
        # wrong:
        # self.weight = self.weight + zeta * self.input.get_delta() * self.output.get_delta() 