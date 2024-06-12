import json_repair
import numpy as np

def extract_polygon(json_str=None, json_dict=None):
    room_info = extract_room_info(json_str,json_dict)
    return [room['floor_polygon'] for room in room_info]

def extract_room_info(json_str=None, json_dict=None):
    '''
    extract polygon, room type and room id when they exist
    '''
    rooms_info = []
    if json_dict is None:
        json_dict = json_repair.loads(json_str)
    if 'rooms' not in json_dict.keys():
        return None
    for room in json_dict['rooms']:
        room_d = {}
        if 'floor_polygon' in room.keys():
            vertices = room['floor_polygon']
            polygon = []
            for vertix in vertices:
                if 'x' in vertix.keys() and 'z' in vertix.keys():
                    polygon.append([vertix['x'],vertix['z']])
            room_d['floor_polygon'] = polygon
        if 'room_type' in room.keys():
            room_d['room_type'] = room['room_type']
        if 'id' in room.keys():
            room_d['id'] = room['id']
        rooms_info.append(room_d)
    return rooms_info

def polygon2bbox(polygon):
    x_max, x_min, y_max, y_min = 0, np.inf, 0, np.inf
    for x,y in polygon:
        x_max = max(x_max,x)
        x_min = min(x_min,x)
        y_max = max(y_max,y)
        y_min = min(y_min,y)
    return (x_min, y_min, x_max, y_max)

def bboxes2bubble(bboxes, th=9):
    '''
        bboxes: list of xyxy definitions for each room
    '''
    edges = []
    for u in range(len(bboxes)):
        for v in range(u+1,len(bboxes)):
            if not collide2d(bboxes[u][:4],bboxes[v][:4],th=th): continue
            # uy0, ux0, uy1, ux1 = bboxes[u][:4]
            # vy0, vx0, vy1, vx1 = bboxes[v][:4]
            # uc = (uy0+uy1)/2,(ux0+ux1)/2
            # vc = (vy0+vy1)/2,(vx0+vx1)/2
            # if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
            #     relation = 5 #'surrounding'
            # elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
            #     relation = 4 #'inside'
            # else:
            #     relation = point_box_relation(uc,bboxes[v,:4])
            # edges.append([u,v,relation])
            edges.append([u,v])
            
    edges = np.array(edges,dtype=int)
    return edges

def collide2d(bbox1, bbox2, th=0):
    return not(
        (bbox1[0]-th > bbox2[2]) or
        (bbox1[2]+th < bbox2[0]) or
        (bbox1[1]-th > bbox2[3]) or
        (bbox1[3]+th < bbox2[1])
    )


def get_edit_distance(g1,g2,g1_dict,g2_dict):
    '''
        g1:         graph 1 -- defined by pairs of connected nodes
        g2:         graph 2
        g1_dict:    dictionary containing info on nodes of g1
                    g1_dict['node2room'] = list of room names where idx is room idx
                    g1_dict['node2id']   = list of room idx to 'id' 
    '''
    pass

def procthor2bubble(version=7):
    from datasets import load_from_disk
    from datasets import Dataset, DatasetDict
    ds_path = f'/network/scratch/l/luozhiha/datasets/procthor_data:v{version}'
    dataset = load_from_disk(ds_path)
    modified_data = {}
    for split in ['train','validation','test']:
        modified_split = []
        dset = dataset[split]
        for idx, data in enumerate(dset):
            print(f'{split}: {idx}')
            room_info = extract_room_info(json_dict = data)
            polygons = [room['floor_polygon'] for room in room_info]
            bboxes = [polygon2bbox(pg) for pg in polygons]
            edges = bboxes2bubble(bboxes,th=2)
            data['edges'] = edges.tolist()
            modified_split.append(data)
        modified_data[split] = Dataset.from_list(modified_split)
    modified_data = DatasetDict(modified_data)
    version = 8
    ds_path = f'/network/scratch/l/luozhiha/datasets/procthor_data:v{version}'
    modified_data.save_to_disk(ds_path)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    procthor2bubble(version=7)