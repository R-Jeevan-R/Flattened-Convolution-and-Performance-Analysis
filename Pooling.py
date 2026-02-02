# Average Pooling Operation Implementation

import Conv_Utils as utils # Importing the utility file for convolution parameters

# Pooling Operation in Python of size 2*2 with stride of 2
def Pooling(Output_fmap):
    Pool_out_fmap = [] # Output Feature Map after Pooling
    E = (utils.E - utils.p2) // utils.stride + 1 # Output Height after Pooling
    F = (utils.F - utils.p1) // utils.stride + 1 # Output Width after Pooling

    for n in range(utils.N):  # For each image in the batch
        batch = []
        for m in range(utils.M):  # For each output channel
            channel = []
            for e in range(E): # For each output row
                row = []
                for f in range(F): # For each output column
                    Pool_sum = [] # Initializing sum for Pooling
                    # Performing Pooling Operation
                    for i in range(utils.p1):
                        for j in range(utils.p2):
                            Pool_sum.append(Output_fmap[n][m][utils.stride * e + i][utils.stride * f + j]) # Accumulating sum for Pooling
                    row.append(sum(Pool_sum) / (utils.p1 * utils.p2)) # Average Pooling
                channel.append(row) # Appending the output row to the current output channel
            batch.append(channel) # Appending the output channel to the current batch
        Pool_out_fmap.append(batch) # Appending the batch to the output feature map
    return Pool_out_fmap # Returning the final output feature map after Pooling



