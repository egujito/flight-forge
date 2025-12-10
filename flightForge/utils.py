import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Optional, Union

def func_from_csv(path, x="x", y="y"):
    x_vals = []
    y_vals = []
    
    with open(path, newline='') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            x_vals.append(float(parts[0]))
            y_vals.append(float(parts[1]))

    return interp1d(x_vals, y_vals, kind="linear", fill_value=0, bounds_error=False), x_vals, y_vals

def unit_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v 

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

class ResultField:
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, name: str, unit: str, color: str, x_label: str = "Time (s)"):
        self.x_data = x_data
        self.y_data = y_data
        self.name = name
        self.unit = unit
        self.color = color
        self.x_label = x_label
        self._interpolator = None

        if len(self.x_data) > 1:
            self._interpolator = interp1d(
                self.x_data, y_data, kind='cubic',
                bounds_error=False, fill_value='extrapolate'
            )

    def __call__(self, t: Optional[float] = None) -> Union[float, None]:
        if t is not None:
            return float(self._interpolator(t)) if self._interpolator else 0.0
        self.plot()
        return None

    def plot(self):
        threshold = 1e-4
        significant_indices = np.where(np.abs(self.y_data) > threshold)[0]
        
        x_plot = self.x_data
        y_plot = self.y_data

        # Trim data if mostly zero at the end (only for time-based plots usually)
        if "Time" in self.x_label and len(significant_indices) > 0:
            last_idx = significant_indices[-1]
            if last_idx < len(self.y_data) - 1:
                cut_idx = min(last_idx + 6, len(self.y_data))
                x_plot = self.x_data[:cut_idx]
                y_plot = self.y_data[:cut_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_plot, y_plot, color=self.color, linewidth=2)
        ax.set_xlabel(self.x_label, fontsize=12)
        ax.set_ylabel(f'{self.name} ({self.unit})', fontsize=12)
        ax.set_title(f'{self.name} vs {self.x_label.split(" ")[0]}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout() 
        plt.show()
