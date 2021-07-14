import logging, numpy as np, rpxdock as rp
from rpxdock.app import dock

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

def filter_sscount(xforms, body, **kw):

    kw = rp.Bunch(kw)
    #Check if values are set in kw, otherwise use the default value. Also try to sanitize input and if not able to, revert
    #to the default value.
    if "min_helix_length" in kw.filter_sscount:
        try:
            min_helix_length = kw.filter_sscount["min_helix_length"]
            min_helix_length = int(min_helix_length)
        except:
            log.warning(f"Could not convert {kw.filter_sscount['min_helix_length']} to integer, reverting to default value (4)")
            min_helix_length = 4
    else:
        min_helix_length = 4

    if "min_sheet_length" in kw.filter_sscount:
        try:
            min_sheet_length = kw.filter_sscount["min_sheet_length"]
            min_sheet_length = int(min_sheet_length)
        except:
            log.warning(f"Coult not convert {kw.filter['min_sheet_length']} to integer, reverting to default value (3).")
            min_sheet_length = 3
    else:
        min_sheet_length = 3

    if "min_loop_length" in kw.filter_sscount:
        try:
            min_loop_length = kw.filter_sscount["min_loop_length"]
            min_loop_length = int(min_loop_length)
        except:
            log.warning(f"Could not convert {kw.filter['min_sheet_length']} to integer, reverting to default value (1)")
            min_loop_length = 1
    else:
        min_loop_length = 1

    if "min_element_resis" in kw.filter_sscount:
        try:
            min_element_resis = kw.filter_sscount["min_element_resis"]
            min_element_resis = int(min_element_resis)
        except:
            log.warning(f"Could not convert min_element_resis {kw.filter['min_element_resis']} to integer, reverting to default value (1)")
            min_element_resis = 1
    else:
        min_element_resis = 1

    if "max_dist" in kw.filter_sscount:
        try:
            max_dist = kw.filter_sscount["max_dist"]
            max_dist = int(max_dist)
        except:
            log.warning(f"Could not convert max_dist {kw.filter_sscount['max_dist']} to integer, reverting to default value (9)")
    else:
        max_dist = 9

    if "sstype" in kw.filter_sscount:
        try:
            sstype = kw.filter_sscount["sstype"]
            sstype = str(sstype)
        except:
            log.warning(f"Could not convert sstype {kw.filter_sscount['sstype']} to string, reverting to default value ('EHL')")
            sstype = "EHL"
    else:
        sstype = "EHL"

    if "confidence" in kw.filter_sscount:
        try:
            confidence = kw.filter_sscount["confidence"]
            confidence = bool(confidence)
        except:
            log.warning(f"Could not convert confidence {kw.filter_sscount['confidence']} to boolean, reverting to default value (False)")
            confidence = False
    else:
        confidence = False

    if "min_ss_count" in kw.filter_sscount:
        try:
            min_ss_count = kw.filter_sscount["min_ss_count"]
            min_ss_count = int(min_ss_count)
        except:
            log.warning(f"Could not convert min_ss_count {kw.filter_sscount['min_ss_count']} to integer, reverting to default value (3)")
    else:
        min_ss_count = 3

    if "simple" in kw.filter_sscount:
        try:
            simple = kw.filter_sscount["simple"]
            simple = bool(simple)
        except:
            log.warning(f"Could not convert argument 'simple' {kw.filter_sscount['simple']} to boolean, reverting to default (True)")
            simple = True
    else:
        simple = True

    if "strict" in kw.filter_sscount:
        try:
            strict = kw.filter_sscount["strict"]
            strict = bool(strict)
        except:
            log.warning(f"Could not convert argument 'strict' {kw.filter_sscount['strict']} to boolean, reverting to default value (False)")
    else:
        strict = False

    spec = dock.get_spec(kw.architecture)
    logging.debug(f"sscount filter args:\n"
                  f"confidence : {confidence}\nmin_helix_length : {min_helix_length}\n"
                  f"min_sheet_length : {min_sheet_length}\nmin_loop_length : {min_loop_length}\n"
                  f"max_dist : {max_dist}\nmin_element_resis: {min_element_resis}\nsstype:{sstype}\n"
                  f"min_ss_count: {min_ss_count}\nstrict: {strict}")

    #TODO: Make this work for n-component docking problems
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
        pos1,
        pos2,
        max_dist,
    )
    #map ss to an object
    body1_ss_map, body2_ss_map = secondary_structure_map(), secondary_structure_map()
    body1_ss_map.map_body_ss(body1, min_helix_length, min_sheet_length, min_loop_length)
    body2_ss_map.map_body_ss(body2, min_helix_length, min_sheet_length, min_loop_length)
    bod_len = (body1.asym_body.nres, body2.asym_body.nres)

    sscounts_data = []
    ss_counts = np.zeros(max(len(pos1), len(pos2)))
    ibest = np.array(range(0, len(ss_counts)))
    for i, (lb, ub) in enumerate(lbub):

        #this loses context information.
        body1_res, body2_res = np.unique(pairs[lb:ub][:,0]), np.unique(pairs[lb:ub][:,1])
        #store residues in an SS element 
        temp_result = {
            "A" : {
               "pdb_file" : body1.pdbfile,
               "E" : 0,
               "H" : 0,
               "L" : 0,
               "total_counts" : 0,
               "resis" : [],
               "paired_resis" : []},
            "B" : {
               "pdb_file" : body2.pdbfile,
               "E" : 0,
               "H" : 0,
               "L" : 0,
               "total_counts" : 0,
               "resis" : [],
               "paired_resis" : []},
            "total_count" : 0}
        resis = np.array([])
        for ss_element in body1_ss_map.ss_index:
            start = body1_ss_map.ss_element_start[ss_element]
            end = body1_ss_map.ss_element_end[ss_element]
            ss_type = body1_ss_map.ss_type_assignments[ss_element]
            list_resis = body1_res[(body1_res >= start + 1) & (body1_res <= end + 1)]
            ss_len = len(list_resis)
            if ss_type in sstype and ss_len >= min_element_resis:
                resis = np.append(resis, list_resis)
                temp_result["A"][ss_type] = temp_result["A"][ss_type] + 1

        # get the paired residues
        # Correct the residue numbers to match the input numbering
        paired_resis = np.unique(pairs[lb:ub, 1][np.isin(pairs[lb:ub, 0], resis)])
        resis = resis - (bod_len[0] * (resis / bod_len[0]).astype(int))
        paired_resis = paired_resis - (bod_len[1] * (paired_resis / bod_len[1]).astype(int))
        temp_result["A"]["total_counts"] = temp_result["A"]["H"] + temp_result["A"]["E"] + temp_result["A"]["L"]
        temp_result["A"]["resis"] = resis
        temp_result["A"]["paired_resis"] = paired_resis

        resis = np.array([])
        for ss_element in body2_ss_map.ss_index:
            start = body2_ss_map.ss_element_start[ss_element]
            end = body2_ss_map.ss_element_end[ss_element]
            ss_type = body2_ss_map.ss_type_assignments[ss_element]
            list_resis = body2_res[(body2_res >= start + 1) & (body2_res <= end + 1)]
            ss_len = len(list_resis)
            if ss_type in sstype and ss_len >= min_element_resis:
                resis= np.append(resis, list_resis)
                temp_result["B"][ss_type] = temp_result["B"][ss_type] + 1

        # get the paired residues
        # Correct the residue numbers to match the input numbering
        paired_resis = np.unique(pairs[lb:ub, 0][np.isin(pairs[lb:ub, 1], resis)])
        resis = resis - (bod_len[1] * (resis / bod_len[1]).astype(int))
        paired_resis = paired_resis - (bod_len[0] * (paired_resis / bod_len[0]).astype(int))
        temp_result["B"]["total_counts"] = temp_result["B"]["H"] + temp_result["B"]["E"] + temp_result["B"]["L"]
        temp_result["B"]["resis"] = resis
        temp_result["B"]["paired_resis"] = paired_resis
        temp_result["total_count"] = temp_result["A"]["total_counts"] + temp_result["B"]["total_counts"]

        if strict:
            #Require that, for a residue to count in an SS element, it has to be paired with a residue that is also in an SS element.
            #reset counts
            temp_result["A"]["E"] = 0
            temp_result["A"]["H"] = 0
            temp_result["A"]["L"] = 0
            temp_result["B"]["E"] = 0
            temp_result["B"]["H"] = 0
            temp_result["B"]["L"] = 0
            temp_result["A"]["total_counts"] = 0
            temp_result["B"]["total_counts"] = 0
            temp_result["total_count"] = 0

            #check that the paired residues are also in an SS element.
            resis = np.array([])
            for ss_element in body1_ss_map.ss_index:
                start = body1_ss_map.ss_element_start[ss_element]
                end = body1_ss_map.ss_element_end[ss_element]
                ss_type = body1_ss_map.ss_type_assignments[ss_element]
                body1_resis = temp_result["B"]["paired_resis"]
                list_resis = body1_resis[(body1_resis >= start + 1) & (body1_resis <= end + 1)]
                ss_len = len(list_resis)
                if ss_type in sstype and ss_len >= min_element_resis:
                    resis = np.append(resis, list_resis)
                    temp_result["A"][ss_type] = temp_result["A"][ss_type] + 1
            temp_result["A"]["total_counts"] = temp_result["A"]["H"] + temp_result["A"]["E"] + temp_result["A"]["L"]
            temp_result["A"]["resis"] = resis

            resis = np.array([])
            for ss_element in body2_ss_map.ss_index:
                start = body2_ss_map.ss_element_start[ss_element]
                end = body2_ss_map.ss_element_end[ss_element]
                ss_type = body2_ss_map.ss_type_assignments[ss_element]
                body2_resis = temp_result["A"]["paired_resis"]
                list_resis = body2_resis[(body2_resis >= start + 1) & (body2_resis <= end + 1)]
                ss_len = len(list_resis)
                if ss_type in sstype and ss_len >= min_element_resis:
                    resis = np.append(resis, list_resis)
                    temp_result["B"][ss_type] = temp_result["B"][ss_type] + 1
            temp_result["B"]["total_counts"] = temp_result["B"]["H"] + temp_result["B"]["E"] + temp_result["B"]["L"]
            temp_result["B"]["resis"] = resis
            temp_result["B"]["paired_resis"] = temp_result["A"]["resis"]
            temp_result["A"]["paired_resis"] = temp_result["B"]["resis"]
            temp_result["total_count"] = temp_result["A"]["total_counts"] + temp_result["B"]["total_counts"]
            ss_counts[i] = temp_result["A"]["total_counts"] + temp_result["B"]["total_counts"]
            sscounts_data.append(temp_result)
        else:
            ss_counts[i] = temp_result["A"]["total_counts"] + temp_result["B"]["total_counts"]
            sscounts_data.append(temp_result)

    if confidence:
        bbest = ss_counts >= min_ss_count
        ibest = ibest[bbest]
        return ibest, ss_counts
    elif simple:
        return ibest, ss_counts
    else:
        return ibest, sscounts_data

