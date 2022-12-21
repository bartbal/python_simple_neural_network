'''
File: start_neuron.py
Project: Insmart_v2
File Created: Wednesday, 14th December 2022 12:21:31 pm
Author: Bart van Netburg (b.van.netburg@insmart.nl)
-----
Last Modified: Wednesday, 21st December 2022 11:17:44 am
Modified By: Bart van Netburg (b.van.netburg@insmart.nl>)
-----
Copyright 2022 - 2022 Insmart B.V., Insmart
'''

import sys
import numpy as np
import axon
import neuron as neu

class start_neuron(neu.neuron):
    # the fixed output of this start neuron
    output : float = None

    def __init__(self, output : float, name : str = ""):
        self.output = float(output)
        #FIXME: not sure if last_z should be 0
        self.last_z = 0
        #FIXME: not sure if b should be 0
        self.b = 0
        # set name
        if(name != ""):
            self.name = name
        else:
            try:
                self.name = "start neuron with output "+str(output)
            except:
                self.name = "noNameStartneuron"
        # super().__init__()

    """
    @brief
    get the output of this axon
    """ 
    def get_output(self) -> float:
        return self.output

    """
    @brief
    Overwrite the get last output method.
    Returns the output of this start neuron
    """
    def get_last_output(self) -> float:
        return self.output
        