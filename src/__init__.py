"""Stock Prediction package.

IMPORTANT: mplfinance/matplotlib MUST be imported before torch
on Windows to avoid DLL conflicts. Force it here.
"""

# Force matplotlib/mplfinance to load first (Windows DLL fix)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance
except ImportError:
    pass