from rpxdock.app import dock
import rpxdock.homog as hm
import logging

def filter_body(body, xforms, arch):
   logging.debug(f"\nArchitecture for filtering {arch}\n")
   arch = arch.upper()
   if arch.startswith('C') and len(arch) == 2:
      #Do cx stuff
      xforms = xforms.reshape(-1, 4, 4)
      sym = "C%i" % i if isinstance(arch, int) else arch
      symrot = hm.hrot([0, 0, 1], 360 / int(sym[1:]), degrees=True)
      xsym = symrot @ xforms
      pos1 = xforms
      pos2 = xsym
      body1 = body
      body2 = body

   else:  #get spec to get bodies
      spec = dock.get_spec(arch)
      if len(body) == 2:
         X = xforms.reshape(-1, xforms.shape[-3], 4, 4)
         B = [b.copy_with_sym(spec.nfold[i], spec.axis[i]) for i, b in enumerate(body)]
         body1 = B[0]
         body2 = B[1]
         pos1 = X[:, 0]
         pos2 = X[:, 1]
      else:
         B = body.copy_with_sym(spec.nfold, spec.axis)
         pos1 = xforms.reshape(-1, 4, 4)  #@ body.pos
         pos2 = spec.to_neighbor_olig @ pos1
         body1 = B
         body2 = B

   return body1, body2, pos1, pos2
