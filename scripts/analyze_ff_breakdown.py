import gzip
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np

# Re-run a detailed check on the FF actions for r_a=0.18, y_min=0.15, y_max=1.25
# Let's see the distance distribution of FF
# We can load the sweep_g64b_multiscene logic or evaluate the 165 cases
print("Analyzing breakdown of False Free cases across distance and cause...")

