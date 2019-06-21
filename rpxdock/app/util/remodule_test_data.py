import sys, rpxdock as rp

def alias_module(frm, to):
   for k, v in list(sys.modules.items()):
      if k.startswith(frm):
         sys.modules[k.replace(frm, to)] = v

def fix_respair_score(rps, rpd):
   new = rp.ResPairScore(rps.xbin, rps.keys, rps.score_map, rps.range_map, rps.respair[0],
                         rps.respair[1], rps.rotspace, rpd)
   new.xbin = rps.xbin
   new.rotspace = rps.rotspace
   new.keys = rps.keys
   new.score_map = rps.score_map
   new.range_map = rps.range_map
   new.respair = rps.respair
   new.aaid = rps.aaid
   new.ssid = rps.ssid
   new.rotid = rps.rotid
   new.stub = rps.stub
   new.pdb = rps.pdb
   new.resno = rps.resno
   new.rotchi = rps.rotchi
   new.rotlbl = rps.rotlbl
   new.id2aa = rps.id2aa
   new.id2ss = rps.id2ss
   new.hier_maps = rps.hier_maps
   new.hier_resls = rps.hier_resls
   new.attr = rps.attr
   return new

def fix_small_respairscore():
   rpdat = rp.data.small_respairdat()
   x = rp.data.small_respairscore()
   y = fix_respair_score(x, rpdat)
   rp.dump(y, rp.datadir + "/pairscore10.pickle")

def fix_small_hscore():
   g = rp.data.small_hscore()
   print(type(g.base))
   print(type(g.hier[0]))

def main():
   alias_module('rpxdock', 'sicdock')

   fix_small_respairscore()
   fix_small_hscore()

if __name__ == '__main__':
   main()