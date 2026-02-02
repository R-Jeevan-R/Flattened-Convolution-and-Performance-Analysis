# Flattened Convolution Implementation

import Conv_Utils as utils # Importing the utility file for convolution parameters
from Flattening_Inputs import flatten_input, flatten_kernel # Importing flattening functions

# Flattened Convolution Function
# Flattening + Matrix Multiplication between flattened kernel and flattened input + Reshaping
def Flattened_Convolution():
    Kernel = flatten_kernel() # Flattening the kernel
    Input = flatten_input() # Flattening the input feature map

    # Output Feature map shape will be : (M, N * E * F)
    Output = [] # Initializing the output feature map 
    for i in range(utils.M): # For each output channel
        row = []
        for j in range(utils.N * utils.E * utils.F): # For each output position
            Sum = 0 # Initializing the sum for the current output cell
            # Performing the dot product between flattened kernel and flattened input
            for k in range(utils.C * utils.R * utils.S):
                Sum += Kernel[i][k] * Input[k][j]  # Accumulating the partial sum
            row.append(Sum) # Appending the computed sum to the current output row
        Output.append(row) # Appending the output row to the output feature map
    

    # Reshaping the output to (N, M, E, F)  
    Output_fmap = [] # Initializing the final output feature map list
    for n in range(utils.N): # For each image in the batch
        batch = []
        for m in range(utils.M):  # For each output channel(or Number of filters)
            channel = []
            for e in range(utils.E): # For each output row
                row = []
                for f in range(utils.F): # For each output column
                    row.append(Output[m][(n * utils.E * utils.F) + (e * utils.F + f)]) # Reshaping the output
                channel.append(row) # Appending the output row to the current output channel
            batch.append(channel) # Appending the output channel to the current batch
        Output_fmap.append(batch) # Appending the batch to the output feature map

    return Output_fmap # Returning the final output feature map after Flattened Convolution

    
