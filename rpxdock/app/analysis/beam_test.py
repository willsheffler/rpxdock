from matplotlib import pyplot as plt
from rpxdock.util import load, dump
import numpy as np

def load_data():
   dirn = "/home/sheffler/debug/rpxdock/beam_test/test2/"
   try:
      assert 0
      ibeam, scores, indices = load(dirn + "topN.pickle")
   except:
      base = "make_plugs_hier_sample_test_rpx_0_ibeam_%i.pickle"
      scores, indices, ibeam = list(), list(), list()
      for i in range(6, 27):
         ibeam.append(i)
         *_, idx, scr = load(dirn + base % i)
         assert idx.shape == scr.shape
         order = np.argsort(-scr)[:100000]
         indices.append(idx[order])
         scores.append(scr[order])
      dump((ibeam, scores, indices), dirn + "topN.pickle")
   return ibeam, scores, indices

def main():
   data = load_data()
   ibeams, scores, indices = data
   topidx = indices[-1]
   for i, (ibeam, scr, idx) in enumerate(zip(*data)):
      # print(
      #     f"{ibeam:2} {len(idx):9,}",
      #     " ".join(f"{scr[q]:7.3f}" for q in range(0, 51, 10)),
      # )
      print(f"{ibeam:2} {2 ** ibeam:10,} {np.sum(np.isin(topidx, idx)):3}")

   ibeams = np.array(ibeams, dtype="i")
   plt.figure(figsize=(16, 12))
   labels = list()
   for i in range(7):
      topN = 10**i
      plt.xscale("log")
      recovs = np.array([np.sum(np.isin(topidx[:topN], idx)) / topN for idx in indices])
      plt.plot(2**(26 - ibeams), recovs)
      # plt.scatter(2 ** (26 - ibeams), recovs)
      labels.append("Recov %i" % topN)
   plt.legend(labels)
   plt.show()

if __name__ == "__main__":
   main()
