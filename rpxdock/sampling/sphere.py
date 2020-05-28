import os
import _pickle

path = os.path.dirname(__file__)

def get_sphere_samples(sym=None):
   fnmap = {
      None: "sphere_0.966_17282.pickle",
      "I": "sphere_ics_asu.pickle",
      "O": "sphere_oct_asu.pickle",
      "T": "sphere_tet_asu.pickle",
   }
   fn = os.path.join(path, fnmap[sym])
   with open(fn, "rb") as inp:
      return _pickle.load(inp)
