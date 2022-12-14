import sys
import numpy as np
import axon
import neuron as neu

class start_neuron(neu.neuron):
    # the fixed output of this start neuron
    output : float = None

    def __init__(self, output : float, name : str = ""):
        self.output = float(output)
        
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
        