class Metrics:
    '''
    Class to hold all metrics and their lookup
    --------------------------------------------------------------------------------------------------------------------------
    total_area_self_consistency                     Metric to check if the sum of room areas is equal to the total area
                                                    (percentage_difference)
    polygon_area_self_consistency                   Metric to check if the computed area of a room polygon is equal to the
                                                    stated area.
                                                    This metric also returns a boolean indicating if the sorted vertex area
                                                    is equal to the unsorted vertex area.
                                                    (avg_percentage_difference, is_sorted_equals_unsorted_area)
    room_count_self_consistency                     Metric to check if the room count is equal to the number of rooms in the
                                                    floorplan. (boolean)
    room_id_self_consistency                        Metric to check if there are no duplicate room ids in the floorplan.
                                                    (boolean)
    polygon_overlap_count_self_consistency          Metric to count the number of overlaps between rooms in the floorplan.
                                                    We would not doulbe count the rooms. (int)
    polygon_containment_count_self_consistency      Metric to count the number of containments between rooms in the floorplan.
                                                    We would not doulbe count the rooms. (int)
    room_height_self_consistency                    Metric to check if the height of a room in the floorplan is equal to the
                                                    height stated. (avg_percentage_difference)
    room_width_self_consistency                     Metric to check if the width of a room in the floorplan is equal to the
                                                    width stated. (avg_percentage_difference)
    num_room_prompt_consistency                     Metric to check if the number of rooms in the floorplan is equal to the
                                                    number of rooms in the prompt. (avg_percentage_difference)
    room_id_prompt_consistency                      Metric to compute the precision and recall of room ids in the floorplan
                                                    with respect to the prompt. Note: I don't think the precision value is
                                                    useful. (precision, recall)
    room_area_prompt_consistency                    Metric to check if the polygon area of a room in the floorplan is equal
                                                    to the area of the room in the prompt. (avg_percentage_difference)
    polygon_area_sum_vs_total_area_prompt_consistency Metric to check if the sum of room polygon areas is equal to the total area
                                                    in the prompt. (percentage_difference)
    room_type_prompt_consistency                    Metric to compute the precision and recall of room types in the
                                                    floorplan with respect to the prompt. Note: I don't think the precision
                                                    value is useful. (precision, recall)
    room_id_type_match_prompt_consistency           Metric to check if the room type of a room in the floorplan is equal to
                                                    the room type of the room in the prompt. (percentage_match)
    room_height_prompot_consistency                 Metric to check if the height of a room in the floorplan is equal to the
                                                    height in the prompt. (avg_percentage_difference)
    room_width_prompot_consistency                  Metric to check if the width of a room in the floorplan is equal to the
                                                    width in the prompt. (avg_percentage_difference)
    json_file_consistency                           Metric to check if the floorplan is valid according to the schema.
                                                    (boolean)
    json_strict_file_consistency                    Metric to check if the floorplan is valid according to the schema in
                                                    strict mode. (boolean)
    --------------------------------------------------------------------------------------------------------------------------
    
    '''
    def __init__(self, metric_names='all'):
        self.metric_lookup = {}
        if 'total_area_self_consistency' in metric_names or metric_names == 'all':
            from .self_consistency import metric_total_area_self_consistency
            self.metric_lookup['total_area_self_consistency'] = metric_total_area_self_consistency
        if 'polygon_area_self_consistency' in metric_names or metric_names == 'all':
            from .self_consistency import metric_polygon_area_self_consistency
            self.metric_lookup['polygon_area_self_consistency'] = metric_polygon_area_self_consistency
        if 'num_room_prompt_consistency' in metric_names or metric_names == 'all':
            from .prompt_consistency import metric_num_room_prompt_consistency
            self.metric_lookup['num_room_prompt_consistency'] = metric_num_room_prompt_consistency
        if 'room_id_prompt_consistency' in metric_names or metric_names == 'all':
            from .prompt_consistency import metric_room_id_prompt_consistency
            self.metric_lookup['room_id_prompt_consistency'] = metric_room_id_prompt_consistency
        if 'room_count_self_consistency' in metric_names or metric_names == 'all':
            from .self_consistency import metric_room_count_self_consistency
            self.metric_lookup['room_count_self_consistency'] = metric_room_count_self_consistency
        if 'room_id_self_consistency' in metric_names or metric_names == 'all':
            from .self_consistency import metric_room_id_self_consistency
            self.metric_lookup['room_id_self_consistency'] = metric_room_id_self_consistency
        if 'polygon_overlap_count_self_consistency' in metric_names or metric_names == 'all':
            from .self_consistency import metric_polygon_overlap_count_self_consistency
            self.metric_lookup['polygon_overlap_count_self_consistency'] = metric_polygon_overlap_count_self_consistency
        # if 'polygon_containment_count_self_consistency' in metric_names or metric_names == 'all':
        #     from .self_consistency import metric_polygon_containment_count_self_consistency
            # self.metric_lookup['polygon_containment_count_self_consistency'] = metric_polygon_containment_count_self_consistency
        if 'room_height_self_consistency' in metric_names or metric_names == 'all':
            from .self_consistency import metric_room_height_self_consistency
            self.metric_lookup['room_height_self_consistency'] = metric_room_height_self_consistency
        if 'room_width_self_consistency' in metric_names or metric_names == 'all':
            from .self_consistency import metric_room_width_self_consistency
            self.metric_lookup['room_width_self_consistency'] = metric_room_width_self_consistency
        if 'room_area_prompt_consistency' in metric_names or metric_names == 'all':
            from .prompt_consistency import metric_room_area_prompt_consistency
            self.metric_lookup['room_area_prompt_consistency'] = metric_room_area_prompt_consistency
        if 'polygon_area_sum_vs_total_area_prompt_consistency' in metric_names or metric_names == 'all':
            from .prompt_consistency import metric_polygon_area_sum_vs_total_area_prompt_consistency
            self.metric_lookup['polygon_area_sum_vs_total_area_prompt_consistency'] = metric_polygon_area_sum_vs_total_area_prompt_consistency
        if 'room_type_prompt_consistency' in metric_names or metric_names == 'all':
            from .prompt_consistency import metric_room_type_prompt_consistency
            self.metric_lookup['room_type_prompt_consistency'] = metric_room_type_prompt_consistency
        if 'room_id_type_match_prompt_consistency' in metric_names or metric_names == 'all':
            from .prompt_consistency import metric_room_id_type_match_prompt_consistency
            self.metric_lookup['room_id_type_match_prompt_consistency'] = metric_room_id_type_match_prompt_consistency
        if 'room_height_prompt_consistency' in metric_names or metric_names == 'all':
            from .prompt_consistency import metric_room_height_prompt_consistency
            self.metric_lookup['room_height_prompt_consistency'] = metric_room_height_prompt_consistency
        if 'room_width_prompt_consistency' in metric_names or metric_names == 'all':
            from .prompt_consistency import metric_room_width_prompt_consistency
            self.metric_lookup['room_width_prompt_consistency'] = metric_room_width_prompt_consistency
        if 'json_file_consistency' in metric_names or metric_names == 'all':
            from .file_consistency import metric_json_file_consistency
            self.metric_lookup['json_file_consistency'] = metric_json_file_consistency
        if 'json_strict_file_consistency' in metric_names or metric_names == 'all':
            from .file_consistency import metric_json_strict_file_consistency
            self.metric_lookup['json_strict_file_consistency'] = metric_json_strict_file_consistency

__all__ = ['Metrics']