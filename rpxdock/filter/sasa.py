import logging, numpy as np, rpxdock as rp

log = logging.getLogger(__name__)

def sasa_filter(boy1, body2, pos1, pos2, min_sasa=750, max_sasa=1500, apply=False, function=None, confidence=False, **kw):

    pairs, lbub = rp.bvh.bvh_collect_pairs_vec(
        body1.bvh_cen,
        body2.bvh_cen,
        pos1 @ body1.pos,
        pos2 @ body2.pos,
        max_dist,
    )

    sasa_data = np.zeros(max(len(pos1), len(pos2)))
    for i, (lb, ub) in enumerate(lbub):
        #get the number of unique residues in each half of the interface (strictly speaking this is not necessary but
        # maybe we could do something cool with it later.
        body1_res, body2_res = np.unique(pairs[lb:ub][:,0]), np.unique(pairs[lb:ub][:,1])

        if apply:
            int1_sasa = (len(body1_res) - 14.2198) / 21.522
            int2_sasa = (len(body2_res) - 14.2198) / 21.522

            tot_int_sasa = int1_sasa + int2_sasa
            sasa_data[i] = append(tot_int_sasa)

        else:
            sasa_data[i] = append(len(body1_res) + len(body2_res))
    if reply:
        return [a and b for a, b in zip(sasa_data < min_sasa, sasa_data > max_sasa]
    else:
            return sasa_data