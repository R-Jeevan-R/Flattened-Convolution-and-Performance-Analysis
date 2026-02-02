# Convolution Utility Parameters and Random Data Generation

import random # Random Module to generate Random values for Input feature map and kernel weights

N = 8 # Batch Size
C = 4 # Input fmap Channels
H = 28 # Input fmap Height
W = 28 # Input fmap Width

R = 5 # Kernel Height
S = 5 # Kernel Width
M = 16 # Output Channels (= Number of Filters)

stride = 2 # Stride for Pooling and Convolution

E =  (H - R)//stride + 1 # Output Height
F =  (W - S)//stride + 1 # Output Width

(p1, p2) = (2, 2) # Pooling Size

# Generating Random Input Feature Map with values in [-1.00, 1.00]
Input_fmap = []
for n_idx in range(N):
    batch = []
    for c_idx in range(C):
        channel = []
        for h_idx in range(H):
            row = []
            for w_idx in range(W):
                value = random.randint(-10, 10) / 10.0 # To get random values between -1.00 to 1.00
                row.append(value)
            channel.append(row)
        batch.append(channel)
    Input_fmap.append(batch)

# Generating Random Kernel Weights with values in [-1.00, 1.00]
Kernel = []
for m_idx in range(M):
    filter = []
    for c_idx in range(C):
        channel = []
        for r_idx in range(R):
            row = []
            for s_idx in range(S):
                value = random.randint(-10, 10) / 10.0 # To get random values between -1.00 to 1.00
                row.append(value)
            channel.append(row)
        filter.append(channel)
    Kernel.append(filter)
