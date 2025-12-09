import csv
import numpy as np 
from scipy.interpolate import interp1d

def func_from_csv(path, x="x", y="y", get_arrs=None):

    x = []
    y = []
    
    with open(path, newline='') as f:
            
        for line in f:
            parts = line.strip().split(",")
            
            if len(parts) < 2:
                continue
            
            x.append(float(parts[0]))
            y.append(float(parts[1]))

    
    if get_arrs: 
        return x, y
    return interp1d(x, y, kind="linear", fill_value=0, bounds_error=False)

def unit_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v # Returns unit versor

def compute_vec(m, u):
     return m * u

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
