# Naive Convolution Implementation

import Conv_Utils as utils # Importing the utility file for convolution parameters

def Naive_Convolution():
    Output_fmap = [] # Initializing the output feature map list

    # Performing Naive Convolution Operation
    for n in range(utils.N):    # For each image in the batch
        batch = []
        for m in range(utils.M):   # For each filter / output channel
            channel = []
            for e in range(utils.E):  # For each output row
                row = []
                for f in range(utils.F): # For each output column
                    Sum = 0 # Initializing the sum for the current output cell
                    # Convolving the filter over the input feature map
                    for c in range(utils.C): 
                        for i in range(utils.R):  
                            for j in range(utils.S):  
                                Sum += utils.Input_fmap[n][c][utils.stride * e + i][utils.stride * f + j] * utils.Kernel[m][c][i][j] # Accumulating the partial sum
                    row.append(Sum) # Appending the computed sum to the current output row
                channel.append(row) # Appending the output row to the current output channel
            batch.append(channel) # Appending the output channel to the current batch
        Output_fmap.append(batch) # Appending the batch to the output feature map
    return Output_fmap # Returning the final output feature map after Convolution
