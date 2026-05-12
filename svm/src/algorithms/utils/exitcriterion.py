import math

class ExitCriterion:
    def __init__(self, maxiter, maxtime, targetaccuracy, loggingfreq):
        self.maxiter = maxiter            # Max #iterations allowed
        self.maxtime = maxtime            # Max execution time allowed
        self.targetaccuracy = targetaccuracy  # Target accuracy to halt algorithm
        self.loggingfreq = loggingfreq    # #datapass between logging

def CheckExitCondition(exitcriterion, currentiter, elapsedtime, measure):
    """
    Check if the given exit criterion has been satisfied. Returns true if satisfied else returns false.
    """
    # A function to determine if it's time to halt execution

    if currentiter >= exitcriterion.maxiter:
        return True
    elif elapsedtime >= exitcriterion.maxtime:
        return True
    elif measure <= exitcriterion.targetaccuracy:
        return True
    elif math.isnan(measure):
        return True

    return False
