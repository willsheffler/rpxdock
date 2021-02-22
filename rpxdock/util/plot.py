import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def coplot(data, show=True):
   try:
      g = sns.PairGrid(data)
   except (IndexError, AttributeError):
      g = sns.PairGrid(pd.DataFrame(data))
   g.map_diag(sns.histplot)
   g.map_offdiag(sns.scatterplot)
   if show: plt.show()
   return g
