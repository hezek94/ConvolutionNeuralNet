def prediction(network, input):
    output = input
    
    for layer in network:
        output = layer.ForwardPro(output)
    return output
        
def train(network, errLoss, primeLoss, x_train, y_train, epoch = 1000, learningRate = 0.01, verbose = True):
    for e in range(epoch):
        error = 0
        for x,y in zip(x_train, y_train):
            output = prediction(network, x)
            
            error += errLoss(y, output)
            
            gradient = primeLoss(y, output)
            
            for reversedlayer in reversed(network):
                gradient = reversedlayer.BackWardPro(gradient, learningRate)
                
        error = error / len(x_train)
        
        if verbose:
            print(f"{e + 1}/{epoch}, error={error}")