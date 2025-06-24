import os

import matplotlib.pyplot as plt

# Use custom matplotlib style
style = os.path.dirname(os.path.abspath(__file__)) + "/style.mplstyle"
plt.style.use(style)
