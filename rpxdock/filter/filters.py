from yaml import load
from yaml import FullLoader as Loader
from rpxdock import filter as rpx_filter
from willutil import Bunch
import rpxdock as rp
import logging

def intersection(l1, l2):
   l3 = [v for v in l1 if v in l2]
   return l3

def filter(xforms, body, **kw):
   #if len(xforms == 0):
   #    logging.warning(f"No transforms provided for filtering")
   #    return None
   #else:
   kw = Bunch(kw, _strict=False)
   #try:
   with open(kw.filter_config) as ff:  #"/home/quintond/git/rpxdock/filters.yml") as ff:
      all_filter_data = load(ff, Loader=Loader)

   all_filter_data = Bunch(all_filter_data)

   #TODO: ibest should be updated with each loop to be the union of the previous ibest and the updated ibest
   #TODO: Make sure all filters have a standardized output in the form extra and ibest.
   extra = Bunch()

   for i, filt in enumerate(all_filter_data.keys()):
      filt_data = Bunch(all_filter_data[filt])
      function = filt_data["type"]
      logging.debug(f"Applying filter {i} of {len(all_filter_data.keys())}: {function}")
      module = function.split("_")[1]
      logging.debug(f"{dir(rpx_filter)}")
      filt_function = getattr(
         getattr(rpx_filter, module), function
      )  #this assumes that the function to call in the filter has the same name as the module

      kw[function] = filt_data

      tmp_ibest, extra[filt] = filt_function(xforms, body, **kw)
      if i == 0:
         ibest = tmp_ibest
      else:
         ibest = intersection(ibest, tmp_ibest)

      logging.debug(f"Extra for {function}: {extra[filt]}")
      logging.debug(f"ibest length {len(ibest)}")
      logging.debug(f"{ibest}")

   if len(ibest) == 0:  #handle the case where filtering removes all docks
      # TODO Generate a log file and abort docking
      print("Filters removed all docks, dumping log file")
      data = dict(
         attrs=dict(arg=kw, filters=all_filter_data, output_prefix=kw.output_prefix,
                    output_body='all'),
         xforms=(["model", "comp", "hrow", "hcol"], xforms),
      )

      for k, v in extra.items():
         if not isinstance(v, (list, tuple)) or len(v) > 3:
            v = ['model'], v
         data[k] = v

      default_label = [f'comp{c}' for c in 'ABCDEFD'[:len(body)]]

      result = rp.Result(
         body_=None if kw.dont_store_body_in_results else body,
         body_label_=[] if kw.dont_store_body_in_results else default_label,
         **data,
      )
      rp.util.dump(result, kw.output_prefix + '_Filter_ERROR_Log.pickle')

      for filt in all_filter_data.keys():
         extra[filt] = extra[filt][slice(0)]

      return slice(0), extra

   else:
      #After all filters have modified ibest, need to reshape extra
      for filt in all_filter_data.keys():
         logging.debug(f"Dumping filter data for {filt} with mask {ibest}")
         logging.debug(f"{filt} contains {extra[filt]}")
         extra[filt] = extra[filt][ibest]
      return ibest, extra

   #except:
   #logging.debug(f"Could not find config file {kw.filter_config}")
   #return list(range(0,len(xforms))), Bunch()

"""
    def filter_redundancy(xforms, body, scores=None, categories=None, every_nth=10, **kw):

    def filter_sscount(body1, body2, pos1, pos2, min_helix_length=4, min_sheet_length=3, min_loop_length=1, min_element_resis=1, max_dist=9.2,
                   sstype="EHL", confidence=0, min_ss_count=3, simple=True, strict=False, **kw):
"""
