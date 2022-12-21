'''
File: main.py
Project: Insmart_v2
File Created: Wednesday, 14th December 2022 12:21:31 pm
Author: Bart van Netburg (b.van.netburg@insmart.nl)
-----
Last Modified: Wednesday, 21st December 2022 10:57:26 pm
Modified By: Bart van Netburg (b.van.netburg@insmart.nl>)
-----
Copyright 2022 - 2022 Insmart B.V., Insmart
'''

import network_manager as nm

def create_learn_and_save_network():
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

    n_manager.save_to_json()

#load network from file
n_manager = nm.network_manager()

n_manager.full_print()

result = n_manager.get_output()
print(result)
