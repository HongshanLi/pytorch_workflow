# Define TrainLogger and ValLogger here
# Logger records the loss for each step and 
# print loss or validation for every predefined 
# number of steps

# define couple of helper funtions here 
# so that every logger can be defined using 
# a few helper functions


# Helpers for classification related tasks
def _compute_accuracy(*args, **kwargs):
    return

def _top_one_error():
    return

def _top_five_error():
    return

# Helpers for regression related tasks
def _compute_error():
    return


class Compose(object):
    def __init__(self, hooks):
        self.hooks = hooks
    def __call__(self, x):
        for hook in hooks:
            x = hook(x)
        return x
    def __repr__(self):
        return


class TrainLogger(object):
    pass

class ValLogger(object):
    pass

class Checkpoint(object):
    pass
