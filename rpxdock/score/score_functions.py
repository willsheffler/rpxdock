import numpy as np
import rpxdock as rp

def score_fun2(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.mean(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.mean(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      side1_stdev = np.std(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2_stdev = np.std(ressc2[lbub1[i, 0]:lbub1[i, 1]])
         # TODO: maybe do this a different way?
      mscore = (side1 + side2) / 2
      sscore = np.sqrt(side1_stdev**2 + side2_stdev**2)
      if np.isnan(sscore):
         sscore = 100

   #ncont_score = a * np.exp( -((ncont) - b)**2 / (2*c**2) )
      a=300
      mu = 75
      sigma = 50
      b=np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
      c=np.log(1 + (sigma**2 / mu**2))
      #ncont_score = a * np.exp( -((ncont) - mu)**2 / (2*sigma**2) )
      if (ub - lb) > 0:
         ncont_score = ( a  / (c * np.sqrt(2 * np.pi) * (ub - lb)) ) * np.exp( -(np.log(ub - lb) - b)**2 / (2*c**2) )
         scores[i] = kw.wts.rpx * mscore + ncont_score
      else:
         scores[i] = 0
   return scores
   
def sasa_priority(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.mean(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.mean(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      side1_stdev = np.std(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2_stdev = np.std(ressc2[lbub1[i, 0]:lbub1[i, 1]])
         # TODO: maybe do this a different way?
      mscore = (side1 + side2) / 2
      sscore = np.sqrt(side1_stdev**2 + side2_stdev**2)
      if np.isnan(sscore):
         sscore = 100

   #ncont_score = a * np.exp( -((ncont) - b)**2 / (2*c**2) )
      a = kw.wts.ncontact
      mu = kw.wts.sasa / 21.522
      if not kw.wts.sasa_error:
        sigma = mu * 0.2433619617913417
      else:
        sigma = kw.wts.sasa_error / 21.522
      b=np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
      c=np.log(1 + (sigma**2 / mu**2))
      #ncont_score = a * np.exp( -((ncont) - mu)**2 / (2*sigma**2) )
      if (ub - lb) > 0:
         ncont_score = ( a  / (c * np.sqrt(2 * np.pi) * (ub - lb)) ) * np.exp( -(np.log(ub - lb) - b)**2 / (2*c**2) )
         scores[i] = kw.wts.rpx * mscore + ncont_score
      else:
         scores[i] = 0
   return scores

def stnd(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.sum(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.sum(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = kw.wts.rpx * mscore + kw.wts.ncontact * (ub - lb)
   return scores
  
def mean(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.mean(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.mean(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = kw.wts.rpx * mscore + kw.wts.ncontact * (ub - lb)
   return scores
  
def median(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.median(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.median(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = kw.wts.rpx * mscore + kw.wts.ncontact * (ub - lb)
   return scores
  
def exp(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.sum(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.sum(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = mscore - ( 4.6679 * ((ub - lb)**0.588  ))
   return scores
  
def lin(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, **kw):
   kw = rp.Bunch(kw)
   scores = np.zeros(max(len(pos1), len(pos2)))
   for i, (lb, ub) in enumerate(lbub):
      side1 = np.sum(ressc1[lbub1[i, 0]:lbub1[i, 1]])
      side2 = np.sum(ressc2[lbub2[i, 0]:lbub2[i, 1]])
      mscore = (side1 + side2) / 2
      scores[i] = mscore - ( 0.7514*(ub - lb) )
   return scores
