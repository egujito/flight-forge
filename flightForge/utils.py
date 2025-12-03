import csv
import numpy as np 
from scipy.interpolate import interp1d

def func_from_csv(path, x="x", y="y"):
    x = []
    y = []
    with open(path, newline='') as f:
            
        for line in f:
            # Remove newline and split by comma
            parts = line.strip().split(",")
            
            # Skip empty or malformed lines
            if len(parts) < 2:
                continue
            
            # Convert to float
            x.append(float(parts[0]))
            y.append(float(parts[1]))

    
    return interp1d(x, y, kind="linear", fill_value=0, bounds_error=False)

def unit_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v # Returns unit versor

def compute_vec(m, u):
     return m * u