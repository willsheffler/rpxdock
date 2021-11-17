import logging, numpy as np, rpxdock as rp
from rpxdock.app import dock

log = logging.getLogger(__name__)

class quasi_body():
   def __init__(self):
      self.pdb = ""
      self.resis_int1 = []
      self.resis_int2 = []
      self.overlap_int1 = 0
      self.overlap_int2 = 0
      self.is_quasi = False
      self.offset = 0
      self.b_len = 0

   def assign_body(self, body, is_quasi=False):
      self.pdb = body.pdbfile
      self.is_quasi = is_quasi
      self.b_len = body.asym_body.nres
      if is_quasi:
         self.offset = int(self.b_len / 2)

   def int1_resis(self, resis):
      # TODO Check that this logic works, it should result in populating residues only if they are below the quasi cutoff, unless both residues are in the same interface
      if self.is_quasi:
         for resi in resis:
            while resi > self.b_len:  #resi could belong to any of resi*n-fold. this corrects for that so resi belongs to the base chain
               resi = resi - self.b_len
            if (resi > self.offset) & (resi - self.offset not in resis):
               self.resis_int1.append(resi - self.offset)
            elif (resi > self.offset) & (resi - self.offset in resis):
               self.resis_int1.append(resi)
            elif (resi < self.offset):
               self.resis_int1.append(resi)
      else:
         for resi in resis:
            while resi > self.b_len:  # resi could belong to any of resi*n-fold. this corrects for that so resi belongs to the base chain
               resi = resi - self.b_len
            self.resis_int1.append(resi)

   def int2_resis(self, resis):
      # TODO Check that this logic works, it should result in populating residues only if they are below the quasi cutoff, unless both residues are in the same interface
      if self.is_quasi:
         for resi in resis:
            while resi > self.b_len:  #resi could belong to any of resi*n-fold. this corrects for that so resi belongs to the base chain
               resi = resi - self.b_len
            if (resi > self.offset) & (resi - self.offset not in resis):
               self.resis_int2.append(resi - self.offset)
            elif (resi > self.offset) & (resi - self.offset in resis):
               self.resis_int2.append(resi)
            elif (resi < self.offset):
               self.resis_int2.append(resi)
      else:
         for resi in resis:
            while resi > self.b_len:  # resi could belong to any of resi*n-fold. this corrects for that so resi belongs to the base chain
               resi = resi - self.b_len
            self.resis_int2.append(resi)

   def isect(self, l1, l2):
      l3 = [v for v in l1 if v in l2]
      return l3

   def intersection(self):
      overlap = self.isect(self.resis_int1, self.resis_int2)
      if len(self.resis_int1) > 0:
         self.overlap_int1 = len(overlap) / len(self.resis_int1)
      else:
         self.overlap_int1 = 0
         print("No residues in interface 1")
      if len(self.resis_int2) > 0:
         self.overlap_int2 = len(overlap) / len(self.resis_int2)
      else:
         self.overlap_int2 = 0
         print("No residues in interface 2")

class quasi_data():
   def __init__(self):
      self.b1 = quasi_body()
      self.b2 = quasi_body()
      self.int1_sasa = 0
      self.int2_sasa = 0
      self.int1_quasi_overlap = 0
      self.int2_quasi_overlap = 0

   def populate_bodies(self, body1, body2, is_quasi=1):
      if is_quasi == 0:
         self.b1.assign_body(body1, is_quasi=True)
         self.b2.assign_body(body2, is_quasi=False)
      elif is_quasi == 1:
         self.b1.assign_body(body1, is_quasi=False)
         self.b2.assign_body(body2, is_quasi=True)

   def int_resis(self, b1_resis, b2_resis, interface=0):
      if interface == 0:
         self.b1.int1_resis(b1_resis)
         self.b2.int1_resis(b2_resis)
      else:
         self.b1.int2_resis(b1_resis)
         self.b2.int2_resis(b2_resis)

   def set_sasa(self, sasa, interface):
      if interface == 0:
         self.int1_sasa = sasa
      else:
         self.int2_sasa = sasa

   def intersect(self):
      self.b1.intersection()
      self.b2.intersection()

   def to_dict(self):
      out_dict = {
         "Body1": {
            "pdb_file": self.b1.pdb,
            "resis_int1": self.b1.resis_int1,
            "resis_int2": self.b1.resis_int2,
            "overlap_int1": self.b1.overlap_int1,
            "overlap_int2": self.b1.overlap_int2,
            "is_quasi": self.b1.is_quasi
         },
         "Body2": {
            "pdb_file": self.b2.pdb,
            "resis_int1": self.b2.resis_int1,
            "resis_int2": self.b2.resis_int2,
            "overlap_int1": self.b2.overlap_int1,
            "overlap_int2": self.b2.overlap_int2,
            "is_quasi": self.b2.is_quasi
         },
         "SASA_Int1": self.int1_sasa,
         "SASA_Int2": self.int2_sasa
      }
      return out_dict

   def quasi_overlap(self):
      for body in (self.b1, self.b2):
         if body.is_quasi:
            self.int1_quasi_overlap = body.overlap_int1
            self.int2_quasi_overlap = body.overlap_int2
            return (body.overlap_int1, body.overlap_int2)

def filter_quasi(xforms, body, **kw):
   kw = rp.Bunch(kw)

   if "min_sasa_int1" in kw.filter_quasi:
      try:
         min_sasa_int1 = kw.filter_quasi["min_sasa_int1"]
         min_sasa_int1 = float(min_sasa_int1)
      except:
         log.warning(
            f"Could not convert min sasa int1 {kw.filter_quasi['min_sasa_int1']} to float, reverting to default value (750)"
         )
         min_sasa_int1 = 375
   else:
      min_sasa_int1 = 375

   if "min_sasa_int2" in kw.filter_quasi:
      try:
         min_sasa_int2 = kw.filter_quasi["min_sasa_int2"]
         min_sasa_int2 = float(min_sasa_int2)
      except:
         log.warning(
            f"Could not convert min sasa int2 {kw.filter_quasi['min_sasa_int2']} to float, reverting to default value (750)"
         )
         min_sasa_int2 = 375
   else:
      min_sasa_int2 = 375

   if "max_sasa_int1" in kw.filter_quasi:
      try:
         max_sasa_int1 = kw.filter_quasi["max_sasa_int1"]
         max_sasa_int1 = float(max_sasa_int1)
      except:
         log.warning(
            f"Could not convert max_sasa_int2 {kw.filter_quasi['max_sasa_int1']} to float, reverting to default value (750)"
         )
         max_sasa_int1 = 1500
   else:
      max_sasa_int1 = 1500

   if "max_sasa_int2" in kw.filter_quasi:
      try:
         max_sasa_int2 = kw.filter_quasi["max_sasa_int2"]
         max_sasa_int2 = float(max_sasa_int2)
      except:
         log.warning(
            f"Could not convert max_sasa_int2 {kw.filter_quasi['max_sasa_int2']} to float, reverting to default value (750)"
         )
         max_sasa_int2 = 1500
   else:
      max_sasa_int2 = 1500

   if "overlap_int1" in kw.filter_quasi:
      try:
         overlap_int1 = kw.filter_quasi["overlap_int1"]
         overlap_int1 = float(overlap_int1)
      except:
         log.warning(
            f"Could not convert overlap_int1 {kw.filter_quasi['overlap_int1']} to float, reverting to default value (0.5)"
         )
         overlap_int1 = 0.5
   else:
      overlap_int1 = 0.5

   if "overlap_int2" in kw.filter_quasi:
      try:
         overlap_int2 = kw.filter_quasi["overlap_int2"]
         overlap_int2 = float(overlap_int2)
      except:
         log.warning(
            f"Could not convert overlap_int2 {kw.filter_quasi['overlap_int2']} to float, reverting to default value (0.5)"
         )
         overlap_int2 = 0.5
   else:
      overlap_int2 = 0.5

   if "max_dist" in kw.filter_quasi:
      try:
         max_dist = kw.filter_quasi["max_dist"]
         max_dist = int(max_dist)
      except:
         log.warning(
            f"Could not convert max_dist {kw.filter_quasi['max_dist']} to int, reverting to default value (9)"
         )
         max_dist = 9
   else:
      max_dist = 9

   if "quasi_comp" in kw.filter_quasi:
      try:
         quasi_comp = kw.filter_sasa["quasi_comp"]
         quasi_comp = int(quasi_comp)
      except:
         log.warning(
            f"Could not convert quasi_comp {kw.filter_quasi['quasi_comp']} to int, reverting to default value (1 (B componentt))"
         )
         quasi_comp = 1
   else:
      quasi_comp = 1

   if "confidence" in kw.filter_quasi:
      try:
         confidence = kw.filter_quasi["confidence"]
         confidence = bool(confidence)
      except:
         log.warning(
            f"Could not convert confidence {kw.filter_quasi['confidence']} to boolean, reverting to default value (False)"
         )
         confidence = False
   else:
      confidence = False

   if "detailed" in kw.filter_quasi:
      try:
         detailed = kw.filter_quasi["detailed"]
         detailed = bool(detailed)
      except:
         log.warning(
            f"Could not convert detailed {kw.filter_quasi['detailed']} to boolean, reverting to default value (False)"
         )
         detailed = False
   else:
      detailed = False

   spec = dock.get_spec(kw.architecture)

   X = xforms.reshape(-1, xforms.shape[-3], 4, 4)
   B = [b.copy_with_sym(spec.nfold[i], spec.axis[i]) for i, b in enumerate(body)]
   body1_int1 = B[0]
   body2_int1 = B[1]
   pos1_int1 = X[:, 0]
   pos2_int1 = X[:, 1]

   Xsym = spec.to_neighbor_olig @ X
   pos1_int2 = X[:, quasi_comp].reshape(-1, 4, 4)
   pos2_int2 = Xsym[:, quasi_comp].reshape(-1, 4, 4)
   body1_int2 = B[quasi_comp]
   body2_int2 = B[quasi_comp]

   pairs_int1, lbub_int1 = rp.bvh.bvh_collect_pairs_vec(
      body1_int1.bvh_cen,
      body2_int1.bvh_cen,
      pos1_int1 @ body1_int1.pos,
      pos2_int1 @ body2_int1.pos,
      max_dist,
   )

   pairs_int2, lbub_int2 = rp.bvh.bvh_collect_pairs_vec(
      body1_int2.bvh_cen,
      body2_int2.bvh_cen,
      pos1_int2 @ body1_int2.pos,
      pos2_int2 @ body2_int2.pos,
      max_dist,
   )

   #qdata = np.zeros(max(len(pos1_int1), len(pos2_int1), len(pos1_int2), len(pos2_int2)))
   qdata = []
   passing = []
   ibest = np.array(range(0, max(len(pos1_int1), len(pos2_int1), len(pos1_int2), len(pos2_int2))))

   #check that lbub_int1 and lbub_int2 are the same length
   if len(lbub_int1) == len(lbub_int2):
      for i, (lb1, ub1) in enumerate(lbub_int1):
         lb2, ub2 = lbub_int2[i]
         temp = quasi_data()
         temp.populate_bodies(B[0], B[1], is_quasi=quasi_comp)

         #get the number of unique residues in each half of the interface (strictly speaking this is not necessary but
         # maybe we could do something cool with it later.
         for j, (pairs, (lb, ub)) in enumerate([(pairs_int1, (lb1, ub1)),
                                                (pairs_int2, (lb2, ub2))]):
            body1_res, body2_res = np.unique(pairs[lb:ub][:, 0]), np.unique(pairs[lb:ub][:, 1])
            temp.int_resis(b1_resis=body1_res, b2_resis=body2_res, interface=j)

            ncont_unique = len(body1_res) + len(body2_res)
            tot_int_sasa = (29.1 * ncont_unique) + 282

            temp.set_sasa(tot_int_sasa, j)

         temp.intersect()  #calculate the overlap between interfaces
         if detailed:
            log.debug(f"Adding detailed information to extra")
            temp.quasi_overlap()
            qdata.append(temp.to_dict())
         else:
            log.debug(f"Standard quasi_filter output")
            qdata(temp.quasi_overlap())

         if (temp.int1_sasa > min_sasa_int1) & (temp.int1_sasa < max_sasa_int1) & (
               temp.int2_sasa > min_sasa_int2) & (temp.int2_sasa < max_sasa_int2):
            #filter passes sasa requirements
            overlap = temp.quasi_overlap()
            if (overlap[0] > overlap_int1) & (overlap[1] > overlap_int2):
               #filter passes overlap requirement
               passing.append(True)
            else:
               passing.append(False)
         else:
            passing.append(False)
   qdata = np.array(qdata)
   if confidence:
      print(f"ibest and ibest[passing]: {ibest} \n {ibest[passing]}")
      return ibest[passing], qdata
   else:
      print(qdata)
      print(f"length of qdata is: {len(qdata)}")
      return ibest, qdata
