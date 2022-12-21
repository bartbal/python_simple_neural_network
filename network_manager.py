'''
File: network_manager.py
Project: Insmart_v2
File Created: Wednesday, 21st December 2022 12:03:30 pm
Author: Bart van Netburg (b.van.netburg@insmart.nl)
-----
Last Modified: Wednesday, 21st December 2022 9:04:18 pm
Modified By: Bart van Netburg (b.van.netburg@insmart.nl>)
-----
Copyright 2022 - 2022 Insmart B.V., Insmart
'''

import neuron as neu
import start_neuron as sn
import axon as ax
import json

class network_manager:

    # to store the lairs with neurons in. list[list[neuron]]
    lairs : list = []
    # used by check_for_change to keep track how many times the results were the same
    same_count : int = 1
    # to store the last result in. Is not set by get_output! but is used for check_for_change
    last_result : list = None

    """
    @brief
    Initialize network manager. Builds a neuron network to the given specifications
    @param int lair_neuron_cound
    The amound of neurons per lair
    @param int lair_count
    The amound of lairs
    @param int start_lair_neuron_count
    The amound of start neurons in the first lair. If None, defaults to lair_neuron_count
    @param int end_lair_neuron_count
    The amound of neurons in the last lair. If None, defaults to lair_neuron_count
    """
    def __init__(self, lair_neuron_count : int, lair_count : int, start_lair_neuron_count: int = None, end_lair_neuron_count : int = None) -> None:
        # first lair
        if start_lair_neuron_count == None:
            start_lair_neuron_count = lair_neuron_count
        lair_list = []
        for n in range(start_lair_neuron_count):
            # create and save start neurons
            lair_list.append(sn.start_neuron(0.20, 'L1N'+str(n+1)))
        self.lairs.append(lair_list)

        # middle lairs
        for l in range(lair_count-2):
            lair_list = []      
            for n in range(lair_neuron_count):
                axon_list = []
                for prev_neuron in self.lairs[l]:
                    axon_list.append(ax.axon(prev_neuron))
                lair_list.append(neu.neuron(axon_list, None, 'L'+str(l+2)+'N'+str(n+1)))
            self.lairs.append(lair_list)

        # last lair
        l = len(lair_list)+1
        if end_lair_neuron_count == None:
            end_lair_neuron_count = lair_neuron_count
        lair_list = []
        for n in range(end_lair_neuron_count):
            axon_list = []
            for prev_neuron in self.lairs[-1]:
                ax.axon(prev_neuron)
                axon_list.append(ax.axon(prev_neuron))
            lair_list.append(neu.neuron(axon_list, None, 'L'+str(l)+'N'+str(n+1)))
        self.lairs.append(lair_list)

    """
    @brief
    Set the output for every start neutron of the network
    @details
    The values will be assigned respectively. 
    Make sure the given outputs list is at least as long as there are start neurons.
    All the given outputs that there are to many will ignored
    @param list[float] outputs
    List with the outputs for the start neurons. check details for more info
    """
    def set_start_neuron_output(self, outputs : list) -> None:
        if len(outputs) < len(self.lairs[0]):
            raise Exception("ERROR: given list of outputs is shorter then list of start neurons")

        for i in range(len(self.lairs[0])):
            self.lairs[0][i].output = outputs[i]

    """
    @brief
    Get the outputs of all the last neutrons
    @return list[float]
    A list with all the results of every last neutron in the network
    """
    def get_output(self) -> list:
        results = []
        for last_neuron in self.lairs[-1]:
            results.append(float(last_neuron.get_output()))
        return results

    """
    @brief
    check for changes between given new_result and self.last_result
    @details
    Be aware that this method uses self.last_result and self.same_count.
    It does not reset self.same_count to 0!
    It does not set self.last_result to anything
    @param list[float] new_result
    The new result to compare with self.last_result
    @param int same_limit
    The amound of times it is allowed for the results to be the same
    @return bool
    True if new_result is not the same as self.last_result,
    False if new_result has been the same as self.last_result for same_limit amound of times
    """
    def check_for_change(self, new_result : list, same_limit : int = 100) -> bool:
        if same_limit == None:
            return True
            
        if new_result == self.last_result:
            self.same_count += 1
            if self.same_count >= same_limit:
                print("\n"+str(self.same_count)+" times the same:\n")
                return False
        else:
            self.last_result = new_result
            self.same_count = 0
        
        return True

    """
    @brief
    Do one learning step for the network
    @param list[flat] expected_outputs
    The learn goal. The network will be learned to match these outputs
    @param float learn_step
    The step to learn with
    """
    def learn(self, expected_outputs : list, learn_step : float = 0.1) -> None:
        if(len(expected_outputs) != len(self.lairs[-1])):
            raise Exception("ERROR: expected outputs not same length as end lairs")
        for i in range(len(self.lairs[-1])):
            self.lairs[-1][i].learn(learn_step, expected_outputs[i])

    """
    @brief
    Learn the neural network to given expected_outputs
    @details
    This method will stop on keyboard interrupt without crashing
    TODO: scale learn_step down when getting closer to goal
    @param list[float] expected_outputs
    A list with the outputs to learn the network too
    @param float learn_step
    The learn_step, how big the step to take in the right direction. to big of a step can overshoot
    @param int same_limit
    If the result of the learning proces has been the same for same_limit times than the program will stop.
    If set to None will always continue.
    @param int print_interval
    The interval amound of cycles to print the results. if set to None, won't print.
    """
    def start_learning(self, expected_outputs : list, learn_step : float = 0.1, same_limit : int = 100, print_interval : int = None) -> None:
        try:
            self.same_count = 1
            self.last_result = []
            count : int = 1
            result : list = []
            while result != expected_outputs and self.check_for_change(result, same_limit):
                self.learn(expected_outputs, learn_step)

                result = self.get_output()
                if print_interval != None:
                    if(count % print_interval == 0):
                        print()
                        print(expected_outputs)
                        print(result)
                        print("\n")
                print("Learning... Cycles: "+str(count), end='\r')
                count += 1
        except KeyboardInterrupt:
            print('\nKeyboardinterrupt on Cycle: '+str(count)+"\n")
            return
        

    """
    @brief
    print all the last results and all the current weightts of every neuron in the network
    @details
    Print structure is:
    <Neuron name>
    last result: <Neuron last result>
    bias: <Neuron bias>
    <Neuron weights>
    ...
    """
    def full_print(self):
        self.lairs[-1][0].chain_print()
        for i in range(len(self.lairs[-1])):
            if i == 0:
                continue
            self.lairs[-1][i].print()
            self.lairs[-1][i].weights_print()

    """
    @brief
    print the name of every neuron in the network
    """
    def print_neurons(self):
        print(self.lairs)
        print("\n------\n")
        for lair in self.lairs:
            for neuron in lair:
                neuron.printName()

    def save_to_jason(self, file_name : str = "NNSave.json"):
        json_save = {}
        for l in range(len(self.lairs)):
            lair = {}
            for n in range(len(self.lairs[l])):
                neuron = {}
                neuron["b"] = self.lairs[l][n].b
                neuron["name"] = self.lairs[l][n].name
                neuron["last_z"] = self.lairs[l][n].last_z
                neuron["last_output"] = self.lairs[l][n].last_output
                in_axons = {}
                for axon in self.lairs[l][n].in_axons:
                    in_axons[axon.input.name[-2:]] = axon.weight
                neuron["in_axons"] = in_axons

                lair["N"+str(n+1)] = neuron
            json_save["L"+str(l+1)] = lair
    
        json_file = json.dumps(json_save)

        f = open(file_name, "w")
        f.write(json_file)
        f.close()

    