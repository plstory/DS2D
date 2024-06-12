from copy import deepcopy
def _compute_recall_precision(TP, FP, FN):
    if len(TP) + len(FN) == 0:
        return None
    precision = len(TP) / (len(TP) + len(FP)) if len(TP) + len(FP) > 0 else 1.0
    recall = len(TP) / (len(TP) + len(FN))
    return precision, recall

def _compute_TP_FP_FN(predicted_set, real_set):
    TP = predicted_set & real_set
    FP = predicted_set - real_set
    FN = real_set - predicted_set
    return TP, FP, FN

def _compute_TP_FP_FN_lists(predicted_L, real_L):
    TP, FP, FN = [], [], []
    real_L = deepcopy(real_L)
    for i, pred in enumerate(predicted_L):
        if pred in real_L:
            TP.append(pred)
            real_L.remove(pred)
        else:
            FP.append(pred)
    FN = real_L
    return TP, FP, FN

def metric_num_room_prompt_consistency(floorplan, prompt_floorplan):
    try:
        prompt_room_count = prompt_floorplan.get_room_count()
    except KeyError:
        return None
    try:
        floorplan_room_count = floorplan.get_room_count()
        return abs(floorplan_room_count - prompt_room_count)/prompt_room_count
    except KeyError:
        return 1.0

def metric_room_id_prompt_consistency(floorplan, prompt_floorplan):
    floorplan_room_ids = floorplan.get_unmodified_room_ids()
    prompt_room_ids = prompt_floorplan.get_unmodified_room_ids()
    TP, FP, FN = _compute_TP_FP_FN(floorplan_room_ids, prompt_room_ids)
    return _compute_recall_precision(TP, FP, FN)

def metric_room_area_prompt_consistency(floorplan, prompt_floorplan):
    floorplan_room_ids = floorplan.get_unmodified_room_ids()
    prompt_room_ids = prompt_floorplan.get_unmodified_room_ids()

    buff = []
    for room_id in floorplan_room_ids & prompt_room_ids:
        try:
            floorplan_room_area = floorplan.get_room_polygon_area(room_id)[0]
            prompt_room_area = prompt_floorplan.get_room_area(room_id)
            buff.append(abs(floorplan_room_area - prompt_room_area) / prompt_room_area)
        except:
            pass
    return sum(buff) / len(buff) if len(buff)>0 else None

def metric_polygon_area_sum_vs_total_area_prompt_consistency(floorplan, prompt_floorplan):
    
    try:
        prompt_total_area = prompt_floorplan.get_total_area()
    except KeyError:
        return None

    floorplan_room_ids = floorplan.get_unmodified_room_ids()
    polygon_total_area = 0.0
    for room_id in floorplan_room_ids:
        try:
            polygon_total_area += floorplan.get_room_polygon_area(room_id)[0]
        except:
            pass
    
    return abs(polygon_total_area - prompt_total_area) / prompt_total_area

def metric_room_type_prompt_consistency(floorplan, prompt_floorplan):

    floorplan_room_types = floorplan.get_room_types()
    prompt_room_types = prompt_floorplan.get_room_types()

    TP, FP, FN = _compute_TP_FP_FN_lists(floorplan_room_types, prompt_room_types)
    return _compute_recall_precision(TP, FP, FN)

def metric_room_id_type_match_prompt_consistency(floorplan, prompt_floorplan):
    floorplan_room_ids = floorplan.get_unmodified_room_ids()
    prompt_room_ids = prompt_floorplan.get_unmodified_room_ids()

    buff, numel = 0, 0
    for room_id in floorplan_room_ids & prompt_room_ids:
        try:
            prompt_room_type = prompt_floorplan.get_room_type(room_id)
        except KeyError:
            continue
        numel += 1
        try:
            floorplan_room_type = floorplan.get_room_type(room_id)
        except KeyError:
            continue
        buff += 1 if floorplan_room_type == prompt_room_type else 0
    return buff / numel if numel > 0 else None

def metric_room_height_prompt_consistency(floorplan, prompt_floorplan):
    floorplan_room_ids = floorplan.get_unmodified_room_ids()
    prompt_room_ids = prompt_floorplan.get_unmodified_room_ids()

    buff = []
    for room_id in floorplan_room_ids & prompt_room_ids:
        try:
            prompt_height = prompt_floorplan.get_room_height(room_id)
            polygon_height = floorplan.get_room_polygon(room_id).height         
            buff.append(abs(polygon_height - prompt_height) / prompt_height)
        except:
            pass
    return sum(buff) / len(buff) if buff else None

def metric_room_width_prompt_consistency(floorplan, prompt_floorplan):
    floorplan_room_ids = floorplan.get_unmodified_room_ids()
    prompt_room_ids = prompt_floorplan.get_unmodified_room_ids()

    buff = []
    for room_id in floorplan_room_ids & prompt_room_ids:
        try:
            prompt_width = prompt_floorplan.get_room_width(room_id)
            polygon_width = floorplan.get_room_polygon(room_id).width        
            buff.append(abs(polygon_width - prompt_width) / prompt_width)
        except:
            pass
    return sum(buff) / len(buff) if buff else None