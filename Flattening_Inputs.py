# Toeplitz Matrix Generation for Convolution Operation

import Conv_Utils as utils # Importing the utility file for convolution parameters 

# Flattening the kernel
def flatten_kernel():
    # Flattened kernel shape will be : (M, C * R * S)
    flatten_kernel = [] # Initializing the flattened kernel list
    for m in range(utils.M): # For each filter
        filters = []
        for c in range(utils.C): # For each kernel(or input) channel
            for r in range(utils.R):   # For each kernel row
                for s in range(utils.S):  # For each kernel column
                    filters.append(utils.Kernel[m][c][r][s])
        flatten_kernel.append(filters)
    return flatten_kernel # Returning the flattened kernel


# Flattening the Input Feature Map
def flatten_input():
    # Flattened input feature map shape will be : (C * R * S, F * E * N)
    flatten_input_fmap = [] # Initializing the flattened input feature map list
    for c in range(utils.C): # For each input channel
        for r in range(utils.R):  # For each kernel row
            for s in range(utils.S): # For each kernel column
                row = []
                for n in range(utils.N): # For each image in the batch  
                    for e in range(utils.E): # For each output height
                        for f in range(utils.F): # For each output width
                            row.append(utils.Input_fmap[n][c][r + utils.stride * e][s + utils.stride * f]) 
                flatten_input_fmap.append(row) 
    return flatten_input_fmap # Returning the flattened input feature map



