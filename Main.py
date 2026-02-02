# Main.py
from Naive_Convolution import Naive_Convolution 
from Flattened_Convolution import Flattened_Convolution    
from Pooling import Pooling
import time


# Taking Command Line Arguments for Perf Analysis 
import sys
if len(sys.argv) > 1:
    MODE = sys.argv[1]
else:
    MODE = "Convolution"  # Default Mode

if __name__ == "__main__":
    if MODE == "Convolution":
        print("Starting Convolution and Pooling Operations...\n")
        print("**************************************************************")
        # Naive Convolution
        print("Performing Naive Convolution...")
        start_time = time.time()
        O_naive = Naive_Convolution()
        end_time = time.time()
        time.sleep(2)  # Just to separate time outputs
        print("Time taken for Naive Convolution: ", end_time - start_time)

        print("\n**************************************************************")
        # Flattened Convolution
        print("Performing Flattened Convolution...")
        start_time = time.time()
        O_flattened = Flattened_Convolution()
        end_time = time.time()
        time.sleep(2)  # Just to separate time outputs
        print("Time taken for Flattened Convolution: ", end_time - start_time)

        print("\n**************************************************************")
        print("Comparing Both Convolution Outputs...")
        time.sleep(2)  # Just to separate outputs in console
        if O_naive == O_flattened:
            print("Naive Convolution and Flattened Convolution outputs are the same.")
        else:
            print("Outputs differ between Naive and Flattened Convolution.")    
        
        print("\n**************************************************************")
        print("Process Completed.")
    elif MODE == 'Pooling':
        print("Starting Pooling Operation on Naive Convolution Output...\n")
        O_naive = Naive_Convolution()  # Getting Naive Convolution Output
        O_flattened = Flattened_Convolution()  # Getting Flattened Convolution Output
        O_pooling_naive = Pooling(O_naive)
        O_pooling_flattened = Pooling(O_flattened)
        if O_pooling_naive == O_pooling_flattened:
            print("Pooling outputs from both Convolution methods are the same.")
        else:
            print("Pooling outputs differ between the two Convolution methods.")    
    elif MODE == "Naive":
        O_Naive = Naive_Convolution()
    elif MODE == "Flattened":
        O_Flattened = Flattened_Convolution()
    else:
        print("Invalid MODE. Please use 'Comparison', 'Naive', or 'Flattened'.") 
        