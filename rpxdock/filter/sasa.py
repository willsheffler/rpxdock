import logging, numpy as np, rpxdock as rp
from rpxdock.app import dock

log = logging.getLogger(__name__)

def filter_sasa(xforms, body, **kw):
    kw = rp.Bunch(kw)

    if "min_sasa" in kw.filter_sasa:
        try:
            min_sasa = kw.filter_sasa["min_sasa"]
            min_sasa = float(min_sasa)
        except:
            log.warning(f"Could not convert min sasa {kw.filter_sasa['min_sasa']} to float, reverting to default value (750)")
            min_sasa = 750
    else:
        min_sasa = 750

    if "max_sasa" in kw.filter_sasa:
        try:
            max_sasa = kw.filter_sasa["max_sasa"]
            max_sasa = float(max_sasa)
        except:
            log.warning(f"Could not convert max_sasa {kw.filter_sasa['max_sasa']} to float, reverting to default value (750)")
            max_sasa = 1500
    else:
        max_sasa = 1500

    if "apply" in kw.filter_sasa:
        try:
            apply = kw.filter_sasa["apply"]
            apply = bool(apply)
        except:
            log.warning(f"Could not convert apply {kw.filter_sasa['apply']} to bool, reverting to default value (True)")
            apply = True
    else:
        apply = True

    if "ncont" in kw.filter_sasa:
        try:
            ncont = kw.filter_sasa["ncont"]
            ncont = bool(ncont)
        except:
            log.warning(f"Could not convert ncont {kw.filter_sasa['ncont']} to bool, reverting to default value (False)")
            ncont = False
    else:
        ncont = False

    if "max_dist" in kw.filter_sasa:
        try:
            max_dist = kw.filter_sasa["max_dist"]
            max_dist = int(max_dist)
        except:
            log.warning(f"Could not convert max_dist {kw.filter_sasa['max_dist']} to int, reverting to default value (9)")
            max_dist = 9
    else:
        max_dist = 9

    if "function" in kw.filter_sasa:
        try:
            function = kw.filter_sasa["function"]
            function = str(function)
        except:
            log.warning(f"Could not convert function {kw.filter_sasa['function']} to string, reverting to default value (None)")
            function = None
    else:
        function = None

    if "confidence" in kw.filter_sasa:
        try:
            confidence = kw.filter_sasa["confidence"]
            confidence = bool(confidence)
        except:
            log.warning(f"Could not convert confidence {kw.filter_sasa['confidence']} to boolean, reverting to default value (False)")
            confidence = False
    else:
        confidence = False

    spec = dock.get_spec(kw.architecture)

    if len(body) == 2:
        X = xforms.reshape(-1, xforms.shape[-3], 4, 4)
        B = [b.copy_with_sym(spec.nfold[i], spec.axis[i]) for i, b in enumerate(body)]
        body1 = B[0]
        body2 = B[1]
        pos1 = X[:,0]
        pos2 = X[:,1]

    else:
        B = body.copy_with_sym(spec.nfold, spec.axis)
        pos1 = xforms.reshape(-1, 4, 4)  #@ body.pos
        pos2 = spec.to_neighbor_olig @ pos1
        body1 = B
        body2 = B

    pairs, lbub = rp.bvh.bvh_collect_pairs_vec(
        body1.bvh_cen,
        body2.bvh_cen,
        pos1 @ body1.pos,
        pos2 @ body2.pos,
        max_dist,
    )

    sasa_data = np.zeros(max(len(pos1), len(pos2)))
    ibest = np.array(range(0, len(sasa_data)))
    for i, (lb, ub) in enumerate(lbub):
        #get the number of unique residues in each half of the interface (strictly speaking this is not necessary but
        # maybe we could do something cool with it later.
        body1_res, body2_res = np.unique(pairs[lb:ub][:,0]), np.unique(pairs[lb:ub][:,1])

        if apply:
            ncont_unique = len(body1_res) + len(body2_res)
            tot_int_sasa = (29.1 * ncont_unique) + 282

            sasa_data[i] = tot_int_sasa

        elif ncont:
            sasa_data[i] = ub - lb
        else:
            sasa_data[i] = len(body1_res) + len(body2_res)
    if confidence:
        return ibest[[a and b for a, b in zip(sasa_data > min_sasa, sasa_data < max_sasa)]], sasa_data
    else:
        return ibest, sasa_data