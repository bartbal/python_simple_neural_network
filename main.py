'''
File: main.py
Project: Insmart_v2
File Created: Wednesday, 14th December 2022 12:21:31 pm
Author: Bart van Netburg (b.van.netburg@insmart.nl)
-----
Last Modified: Wednesday, 21st December 2022 11:24:00 am
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

def test() -> int:
    return 2

# L1
L1N1 = sn.start_neuron(0.20, 'L1N1')
L1N2 = sn.start_neuron(0.45, 'L1N2')

# L2
L1N1_L2N1 = ax.axon(L1N1)
L1N2_L2N1 = ax.axon(L1N2)
L2N1 = neu.neuron(1, [L1N1_L2N1, L1N2_L2N1], 'L2N1')

L1N1_L2N2 = ax.axon(L1N1)
L1N2_L2N2 = ax.axon(L1N2)
L2N2 = neu.neuron(1, [L1N1_L2N2, L1N2_L2N2], 'L2N2')

# L3
L2N1_L3N1 = ax.axon(L2N1)
L2N2_L3N1 = ax.axon(L2N2)
L3N1 = neu.neuron(1, [L2N1_L3N1, L2N2_L3N1], 'L3N1')

L2N1_L3N2 = ax.axon(L2N1)
L2N2_L3N2 = ax.axon(L2N2)
L3N2 = neu.neuron(1, [L2N1_L3N2, L2N2_L3N2], 'L3N2')

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

learnGoal1 = 0.5
learnGoal2 = 0.2

# L3N1.learn(1, learnGoal1)
# L3N2.learn(1, learnGoal2)
# result_L3N1 : float = L3N1.get_output()
# result_L3N2 : float = L3N2.get_output()
# print(result_L3N1)
# print(result_L3N2)
# print("---")

count : int = 1
while (result_L3N1 != learnGoal1) and (result_L3N2 != learnGoal2) and check_for_change(result_L3N1, result_L3N2):
    L3N1.learn(1, learnGoal1)
    L3N2.learn(1, learnGoal2)

    result_L3N1 : float = L3N1.get_output()
    result_L3N2 : float = L3N2.get_output()
    # print(count)
    # print(result_L3N1)
    # print(result_L3N2)
    count += 1
print(result_L3N1)
print(result_L3N2)
print(count)
L3N1.chain_print()
L3N2.print()
L3N2.weights_print()
