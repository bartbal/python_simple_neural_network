'''
File: main.py
Project: Insmart_v2
File Created: Wednesday, 14th December 2022 12:21:31 pm
Author: Bart van Netburg (b.van.netburg@insmart.nl)
-----
Last Modified: Wednesday, 21st December 2022 6:31:25 pm
Modified By: Bart van Netburg (b.van.netburg@insmart.nl>)
-----
Copyright 2022 - 2022 Insmart B.V., Insmart
'''

import sys
import numpy as np
import perceptron as pc
import neuron as neu
import start_neuron as sn
import axon as ax
import network_manager as nm

def manual_network():

    # L1
    L1N1 = sn.start_neuron(0.20, 'L1N1')
    L1N2 = sn.start_neuron(0.45, 'L1N2')

    # L2
    L1N1_L2N1 = ax.axon(L1N1)
    L1N2_L2N1 = ax.axon(L1N2)
    L2N1 = neu.neuron([L1N1_L2N1, L1N2_L2N1], None, 'L2N1')

    L1N1_L2N2 = ax.axon(L1N1)
    L1N2_L2N2 = ax.axon(L1N2)
    L2N2 = neu.neuron([L1N1_L2N2, L1N2_L2N2], None, 'L2N2')

    # L3
    L2N1_L3N1 = ax.axon(L2N1)
    L2N2_L3N1 = ax.axon(L2N2)
    L3N1 = neu.neuron([L2N1_L3N1, L2N2_L3N1], None, 'L3N1')

    L2N1_L3N2 = ax.axon(L2N1)
    L2N2_L3N2 = ax.axon(L2N2)
    L3N2 = neu.neuron([L2N1_L3N2, L2N2_L3N2], None, 'L3N2')

    sameCount : int = 1
    old_result1 : float = 0
    old_result2 : float = 0
    def check_for_change(new_result1 : float, new_result2 : float):
        global sameCount
        global old_result1
        global old_result2
        if new_result1 == old_result1 and new_result2 == new_result2:
            sameCount += 1
            if sameCount >= 10:
                print("\n"+str(sameCount)+" times the same:\n")
                return False
        else:
            old_result1 = new_result1
            old_result2 = old_result2
            sameCount = 0
        
        return True

    L3N1.chain_print()
    L3N2.print()
    L3N2.weights_print()
    result_L3N1 : float = L3N1.get_output()
    result_L3N2 : float = L3N2.get_output()
    print("---")
    print(result_L3N1)
    print(result_L3N2)
    print("---")

    learnGoal1 = 0.8
    learnGoal2 = 0.1

    count : int = 1
    while (result_L3N1 != learnGoal1) and (result_L3N2 != learnGoal2) and check_for_change(result_L3N1, result_L3N2):
        L3N1.learn(0.1, learnGoal1)
        L3N2.learn(0.1, learnGoal2)

        result_L3N1 : float = L3N1.get_output()
        result_L3N2 : float = L3N2.get_output()
        if(count % 5000 == 0):
            print()
            print(result_L3N1)
            print(result_L3N2)
            print("\n")
        print("Learning... Cycles: "+str(count), end='\r')
        count += 1
    print(result_L3N1)
    print(result_L3N2)
    print(count)
    L3N1.chain_print()
    L3N2.print()
    L3N2.weights_print()

# create network manager
n_manager = nm.network_manager(4, 5, 2, 2)

# set the outputs of the start neurons
n_manager.set_start_neuron_output([0.45, 0.50])

# print all the values of the network
n_manager.full_print()

# get the result of forward propagation
result = n_manager.get_output()
print(result)

# learn the network
expected_results = [0.1, 0.8]
n_manager.start_learning(expected_results, 0.1, 500, 100)

# get the result of forward propagation to see what the effect of learning has been
result = n_manager.get_output()
print(result)

# print all the values of the network to see changes done by learning
n_manager.full_print()
