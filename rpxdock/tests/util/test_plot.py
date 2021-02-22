from rpxdock.util.plot import *
import seaborn as sns
import numpy as np
import pandas as pdb

def test_plot():
   data = np.random.normal(0, 1, (100, 4))
   data = pd.DataFrame(data)

   fig = coplot(data, show=False)

if __name__ == '__main__':
   test_plot()
