import numpy as np

class Results:
    def __init__(self):
        self.iterations = []
        self.times = []
        self.optmeasures = []
        self.L = []
        self.L_hat = []

def logresult(results, current_iter, elapsed_time, opt_measure, L_hat=None, L=None):
    """
    Append execution measures to Results.
    """
    results.iterations.append(current_iter)
    results.times.append(elapsed_time)
    results.optmeasures.append(opt_measure)
    if L != None:
        results.L.append(L)
    if L_hat != None:
        results.L_hat.append(L_hat)
    return 
