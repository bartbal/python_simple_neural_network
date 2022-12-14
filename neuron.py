import sys
import numpy as np
import math
import axon
from typing import List, Set, Dict, Tuple


class neuron:
    
    #the bias as float
    b : float = None

    # the name of this neuron as string. Used for debugging
    name : str = ""

    # out_axons: List[axon] = []
    out_axons : list = []

    # The list of input axons
    in_axons : list = []

    # The last Z of this neuron
    last_z : float = None

    # the last output value of this neuron
    last_output : float = None

    # to track if this neuron already learned
    learned : bool = False

    def __init__(self, bias : float, in_axons : list, name : str = ""):
        self.b = float(bias)

        self.in_axons = in_axons
        
        # set name
        if(name != ""):
            self.name = name
        else:
            try:
                self.name = "neuron with bias "+str(bias)
            except:
                self.name = "noNameneuron"

        # set outputs for inputs
        for axon in in_axons:
            axon.set_output(self)

    """
    @brief
    Print the network from this neuron and what comes before
    """
    def chain_print(self):
        if len(self.in_axons) > 0:
            self.in_axons[0].chain_print()
            for axon in self.in_axons[1:]:
                axon.input_print()
        self.print()
        self.weights_print()
    
    """
    @brief
    print the name with th last output value
    """
    def print(self):
        self.printName() 
        print(self.last_output)

    """
    @brief
    Print te weights of the input axons
    """
    def weights_print(self):
        for axon in self.in_axons:
            axon.print()

    """
    @brief
    get the result of this neuron
    """
    def get_output(self) -> float:
        self.learned = False
        sum = 0
        for in_axon in self.in_axons:
            sum += in_axon.get_output()
        sum += self.b

        #update last output and last_z and return it
        self.last_z = sum
        self.last_output = self.sigmoid(sum)
        # print(self.name+": "+str(self.last_output))
        return self.last_output

    """
    @brief
    Add out axon to list of output axons for this neuron
    """
    def add_output(self, target_axon : axon) -> None:
        self.out_axons.append(target_axon)

    """
    @brief
    Print the name given to this neuron
    """
    def printName(self) -> None:
        print(self.name)

    """
    @param float zeta
    the incremental correction.
    @param float expected_out
    the output that was expected
    """
    def learn(self, zeta : float = 0.1, expected_out : float = None) -> None:
        if self.learned:
            return
        #calculate detla
        delta = self.get_delta(expected_out)
        #calculate weights
        for out_axon in self.out_axons:
            out_axon.update_weight(zeta, delta)
        #calculate bias
        self.update_bias(zeta, delta)
        # go to next lair
        for out_axon in self.out_axons:
            # don't have to give it expected_out becouse that is only for the last lair
            out_axon.learn(zeta, None)

    """
    @brief
    returns the last output
    If no forward propagation has been done yet, returns None
    @return float
    The last output of this neuron or None
    """
    def get_last_output(self) -> float:
        return self.last_output

    """
    @brief
    get the delta of this object
    @param float out_delta
    the delta of the next neutron.
    @param float expected_out
    the expected output
    """
    def get_delta(self, expected_out : float = None) -> float:
        if self.learned:
            return self.last_delta
        if expected_out == None:
            # this is the final neuron
            self.last_delta = self.sigmoidDif(self.last_z)*(expected_out-self.get_last_output())
            return self.last_delta
        else:
            # other neurons
            delta_sum = 0
            for out_axon in self.out_axons:
                delta_sum += (out_axon.get_weight() * out_axon.output.get_delta())
            self.last_delta = self.sigmoidDif(self.last_z)*delta_sum
            return self.last_delta

    """
    @brief 
    calculate new bias for this neuron
    @details
    sets the bias of the neuron to the new calculated bias value
    @param float zeta
    the amount to increment the bias with
    @param float delta
    The delta needed for the calculation
    """
    def update_bias(self, zeta : float, delta : float) -> None:
        #bias = zeta * delta
        self.b = self.b+zeta*delta

        # Wrong:
        #weight + zeta * input_delta*output_delta = new_weight
        # self.bias = self.bias + zeta * self.get_delta(expected_out) 

    """
    @brief
    this is a stable sigmoid function
    @source
    https://www.delftstack.com/howto/python/sigmoid-function-python/
    @param float x
    input value to run the sigmoid function on
    @return float
    The result from the sigmoid function
    """
    def sigmoid(self, x : float) -> float:
        # print(self.name+" sigmoid: "+str(x))
        sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
        return sig

    """
    @brief
    derivative of the sigmoid function
    @source
    https://stackoverflow.com/questions/10626134/derivative-of-sigmoid
    @param float x
    input value for the sigmoid function
    @return float
    The result
    """
    def sigmoidDif(self, x : float) -> float:
        f = 1/(1+np.exp(-x))
        df = f * (1 - f)
        return df