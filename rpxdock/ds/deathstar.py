import rpxdock as rp
from willutil import Bunch

class DeathStar(object):
   """represents data for asymmetrized cage"""
   def __init__(self, body, cagesym, cycsym, ifaceclasses, dofclasses, doftypes):
      super(DeathStar, self).__init__()
      self.body = body
      self.cagesym = cagesym
      self.cycsym = cycsym
      self.ifaceclasses = ifaceclasses
      self.dofclasses = dofclasses
      self.doftypes = doftypes

def cage2cyclic(
   cagesym,
   cycsym,
):
   assert 0