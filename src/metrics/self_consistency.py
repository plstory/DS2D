from collections import defaultdict
def metric_room_count_self_consistency(floorplan):
    try:
        return floorplan.get_room_count() == floorplan.get_num_rooms()
    except:
        return None

def metric_room_id_self_consistency(floorplan):
    return len(floorplan.get_room_ids()) == floorplan.get_num_rooms()

def metric_total_area_self_consistency(floorplan):
    room_ids = floorplan.get_room_ids()
    try:    
        total_area = floorplan.get_total_area()
    except:
        return None
    area_diff = total_area
    for room_id in room_ids:
        try:
            area_diff -= floorplan.get_room_area(room_id)
        except:
            try:
                area_diff -= floorplan.get_room_polygon_area(room_id)[0]
            except:
                pass
    return abs(area_diff) / total_area

def metric_polygon_area_self_consistency(floorplan):
    room_ids = floorplan.get_room_ids()
    area_scores = []
    if_align_score = 0 # if area computed with sorted vertices is the same as the area computed with unsorted vertices
    num_valid_rooms = 0
    for room_id in room_ids:
        try:
            computed_area, if_align = floorplan.get_room_polygon_area(room_id)
            if_align_score += 1 if if_align else 0
            stated_area = floorplan.get_room_area(room_id)
            area_scores.append(abs(computed_area - stated_area) / stated_area)
            num_valid_rooms += 1
        except:
            pass

    return (sum(area_scores)/num_valid_rooms, if_align_score / num_valid_rooms) if num_valid_rooms>0 else None


def metric_polygon_overlap_count_self_consistency(floorplan):
    return floorplan.count_room_overlaps() > 0

def metric_polygon_containment_count_self_consistency(floorplan):
    raise NotImplementedError("Not implemented yet")

def metric_room_height_self_consistency(floorplan):

    room_ids = set(floorplan.get_room_ids())

    height_scores = []
    for room_id in room_ids:
        try:
            stated_height = floorplan.get_room_height(room_id)
            polygon_height = floorplan.get_room_polygon(room_id).height
            height_scores.append(abs(polygon_height - stated_height) / stated_height)
        except:
            pass
    return sum(height_scores)/len(height_scores) if height_scores else None

def metric_room_width_self_consistency(floorplan):
    room_ids = set(floorplan.get_room_ids())

    width_scores = []
    for room_id in room_ids:
        try:
            stated_width = floorplan.get_room_width(room_id)
            polygon_width = floorplan.get_room_polygon(room_id).width
            width_scores.append(abs(polygon_width - stated_width) / stated_width)
        except:
            pass
    return sum(width_scores)/len(width_scores) if width_scores else None