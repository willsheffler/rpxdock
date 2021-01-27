import logging, numpy as np, rpxdock as rp

log = logging.getLogger(__name__)


class secondary_structure_map:
    def __init__(self):
        self.ss_index = []
        self.ss_type_assignments = []
        self.ss_element_start = []
        self.ss_element_end = []
        self.ss = []

    def assign_ss(self, index_num, ss_type, start, end):
        self.ss_index.append(index_num)
        self.ss_type_assignments.append(ss_type)
        self.ss_element_start.append(start)
        self.ss_element_end.append(end)

    def set_ss(self, ss):
        self.ss = ss

    def map_body_ss(self, body, min_helix_length=4, min_sheet_length=3, min_loop_length=1):
        ss_params = {"H": min_helix_length, "E": min_sheet_length, "L": min_loop_length}

        self.set_ss(body.ss)

        # first residue assignment
        temp_ss = body.ss[0]
        temp_start = 0
        n = 0
        for i, ss_at_resi in enumerate(body.ss):
            if ss_at_resi == temp_ss:  # still in the same ss element
                temp_end = i
            else:  # left the ss element, record it if it is sufficiently long
                end = temp_end
                start = temp_start
                ss_type = temp_ss
                if end - start >= ss_params[ss_type] - 1:
                    # create an SS element entry
                    self.assign_ss(n, ss_type, start, end)
                    n = n+1
                temp_start = i
                temp_ss = ss_at_resi

def filter_sscount(body1, body2, pos1, pos2, min_helix_length=4, min_sheet_length=3, min_loop_length=1, min_element_resis=1, max_dist=8.0,
                   sstype="EHL", confidence=0, min_ss_count=3, **kw):

    pairs, lbub = rp.bvh.bvh_collect_pairs_vec(
        body1.bvh_cen,
        body2.bvh_cen,
        pos1 @ body1.pos,
        pos2 @ body2.pos,
        max_dist,
    )
    #map ss to an object
    body1_ss_map, body2_ss_map = secondary_structure_map(), secondary_structure_map()
    body1_ss_map.map_body_ss(body1, min_helix_length, min_sheet_length, min_loop_length)
    body2_ss_map.map_body_ss(body2, min_helix_length, min_sheet_length, min_loop_length)

    ss_counts = np.zeros(max(len(pos1), len(pos2)))
    for i, (lb, ub) in enumerate(lbub):
        body1_res, body2_res = np.unique(pairs[lb:ub][:,0]), np.unique(pairs[lb:ub][:,1])
        temp_counts = {"H" : 0, "E" : 0, "L" : 0}
        for ss_element in body1_ss_map.ss_index:
            start = body1_ss_map.ss_element_start[ss_element]
            end = body1_ss_map.ss_element_end[ss_element]
            ss_type = body1_ss_map.ss_type_assignments[ss_element]
            ss_len = len(body1_res[body1_res >= start + 1][body1_res[body1_res >= start + 1] < end + 1])
            if ss_type in sstype and ss_len >= min_element_resis:
                temp_counts[ss_type] = temp_counts[ss_type] + 1
        for ss_element in body2_ss_map.ss_index:
            start = body2_ss_map.ss_element_start[ss_element]
            end = body2_ss_map.ss_element_end[ss_element]
            ss_type = body2_ss_map.ss_type_assignments[ss_element]
            ss_len = len(body2_res[body2_res >= start + 1][body2_res[body2_res >= start + 1] < end + 1])
            if ss_type in sstype and ss_len >= min_element_resis:
                temp_counts[ss_type] = temp_counts[ss_type] + 1
        ss_counts[i] = temp_counts["H"] + temp_counts["E"] + temp_counts["L"]
    if confidence==1:
        return ss_counts >= min_ss_count
    else:
        return ss_counts
