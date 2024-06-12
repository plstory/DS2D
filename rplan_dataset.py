import json
from datasets import load_from_disk, DatasetDict
import datasets
import numpy as np
import random
room_label = {
    0: "LivingRoom",
    1: "MasterRoom", 
    2: "Kitchen", 
    3: "Bathroom",
    4: "DiningRoom",
    5: "ChildRoom",
    6: "StudyRoom",
    7: "SecondRoom",
    8: "GuestRoom", 
    9: "Balcony",
    10: "Entrance",
    11: "Storage", 
    12: "Wall-in", 
    13: "External",
    14: "ExteriorWall",
    15: "FrontDoor", 
    16: "InteriorWall",
    17: "InteriorDoor",
}

def get_custom_dataset(dataset_config, tokenizer, split, testing=False):
    exprm = int(dataset_config.exprm[:1])
    ds_dir = 'datasets/rplan_converted/'
    dd = []
    for idx in [5,6,7,8]:
        if idx == exprm:
            continue
        dd.append(load_from_disk(f'{ds_dir}{idx}'))
    dataset = DatasetDict()
    for key in dd[0]:
        dataset[key] = datasets.concatenate_datasets([ddd[key] for ddd in dd])
    

    if split == 'validation':
        split = 'test'
    dataset = dataset[split]

    pixel2len = 18/256
    pixel2area = pixel2len**2
    
    def process_sample(data):
        if str(dataset_config.exprm).find('new') == -1:
            num_rooms = len(data['rooms'])
            json_str = f'{{"rooms": ['
            for room_idx, room_info in enumerate(data['rooms']):
                json_str += f'{{"room_type": "{room_label[room_info[-2]]}", '
                json_str += '"floor_polygon": ['
                for x,y in data['polygons'][room_idx]:
                    json_str += f'{{"x": {x}, "z": {y}}}, '
                json_str = json_str.strip(', ') + '], '
                json_str += f'"id": "room|{room_idx}"}}, '
            json_str = json_str.strip(', ') + ']}'
        else:
            num_rooms = len(data['rooms'])
            total_area = 0
            room_types = []
            json_str = f'"rooms": ['
            for room_idx, room_info in enumerate(data['rooms']):
                y0,x0,y1,x1,c1,c2,area, height, width = room_info
                total_area += area
                json_str += f'{{"area": {area*pixel2area:.2f}, '
                json_str += f'"room_type": "{room_label[c1]}", '
                room_types.append(room_label[c1])
                json_str += '"floor_polygon": ['
                for x,y in data['polygons'][room_idx]:
                    json_str += f'{{"x": {x}, "z": {y}}}, '
                json_str = json_str.strip(', ') + '], '
                json_str += f'"height": {height*pixel2len:.2f}, '
                json_str += f'"width": {width*pixel2len:.2f}, '
                json_str += f'"id": "room|{room_idx}"}}, '
            json_str = json_str.strip(', ') + ']}'
            json_str = f'{{"room_count": {len(data["rooms"])}, "total_area": {total_area*pixel2area:.2f}, "room_types": {room_types}, ' + json_str
            json_str = json_str.strip(', ')
            json_str = json_str.replace("'",'"')
        
        prompt_d={}
        prompt_d = json.loads(json_str.replace("'",'"'))
        for room_dict in prompt_d['rooms']:
            del room_dict['floor_polygon']
            for k in list(room_dict.keys()):
                if random.random() < 0.5:
                    del room_dict[k]
            if len(room_dict.keys()) == 0:
                del room_dict
        if len(prompt_d['rooms']) == 0:
            del prompt_d['rooms']
        rands = np.random.random(len(prompt_d.keys()))
        rands[np.argmax(rands)] = 1.0
        for idx, k in enumerate(list(prompt_d.keys())):
            if rands[idx] < 0.5:
                del prompt_d[k]

        instruction_str = 'you are to generate a floor plan in a JSON structure. you have to satisfy the adjacency constraints given as pairs of neighboring rooms; two connecting rooms are presented as (room_type1 room_id1, room_type2 room_id2). you also need to satisfy additional contraints given by the user.'
        adjacency_str = f'total number of rooms: {num_rooms}; adjacency pairs: '
        for u,v,_ in data['edges']:
            type_u = room_label[data['rooms'][u][4]]
            type_v = room_label[data['rooms'][v][4]] 
            id_u = f"room|{u}"
            id_v = f"room|{v}"
            adjacency_str += f'({type_u} = "{id_u}", {type_v} = "{id_v}"), '
        adjacency_str = adjacency_str.strip(', ')
        user_str = adjacency_str
        
        if len(prompt_d.keys())>0:
            user_str += f'. additional constraints: {str(prompt_d)}'

        prompt_str = f"""<|start_header_id|>system<|end_header_id|> {instruction_str}<|eot_id|><|start_header_id|>user<|end_header_id|> {user_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|> """
        prompt = tokenizer(f"{tokenizer.bos_token}{prompt_str}", add_special_tokens=False)
        floorplan = tokenizer(f"{json_str}{tokenizer.eos_token}", add_special_tokens=False)
        
        input_ids = prompt['input_ids'] + floorplan['input_ids']
        attention_mask = [1] * (len(prompt['input_ids']) + len(floorplan['input_ids']))
        labels = [-100] * len(prompt['input_ids']) + floorplan['input_ids']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return dataset.map(
        process_sample, 
        remove_columns=list(dataset.features)
    )

if __name__ == '__main__':
    get_custom_dataset({'exprm':4}, None, 'train')