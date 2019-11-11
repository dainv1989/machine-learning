def initialize_parameters_deep(layers_dims):
    ...
    return parameters 

def L_model_forward(X, parameters):
    ...
    return AL, caches

def compute_cost(AL, Y):
    ...
    return cost

def L_model_backward(AL, Y, caches):
    ...
    return grads

def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
