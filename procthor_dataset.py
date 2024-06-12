import json
from datasets import load_from_disk
import numpy as np
import random
from copy import deepcopy

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
    version = dataset_config.ds_version
    ds_path = 'datasets/procthor_converted'
    dataset = load_from_disk(ds_path)
    dataset = dataset[split] 
    exclude_idx = []

    #process dataset according to experiments needed
    def process_sample(sample):
        sample = deepcopy(sample)
        exprm = dataset_config.exprm
        if exprm.startswith('mask'):
            for room_dict in sample['prompt']['rooms']:
                for k in list(room_dict.keys()):
                    if random.random() < 0.5:
                        del room_dict[k]
                if len(list(room_dict.keys())) == 0:
                    del room_dict
            if len(sample['prompt']['rooms']) == 0:
                del sample['prompt']['rooms']
            rands = np.random.random(len(sample['prompt'].keys()))
            rands[np.argmax(rands)] = 1.0
            for idx, k in enumerate(list(sample['prompt'].keys())):
                if rands[idx] < 0.5:
                    del sample['prompt'][k]
        
        if exprm.startswith('preset_mask'):
            prompt = {}
            prompt['room_count'] = sample['room_count']
            prompt['total_area'] = sample['total_area']
            partial_prompt = np.random.randint(5)
            
            if partial_prompt in [1,3]: # only_total_area
                prompt['total_area'] = sample['prompt']['total_area']
            prompt['room_types'] = sample['room_types']
            if partial_prompt in [2,3,4]: # only_room_area
                rooms = deepcopy(sample['rooms'])
                if partial_prompt in [3,4]:
                    rands = np.random.random(len(rooms))
                    rands[np.argmax(rands)] = 1.0
                    drop_idx = np.where(rands<0.5)[0]
                    for idx in sorted(drop_idx,reverse=True):
                        del rooms[idx]
                for room in rooms:
                    for key in list(room.keys()):
                        if key not in set(['area','room_type', 'id']):
                            del room[key]
                prompt['rooms'] = rooms
            sample['prompt'] = prompt
            
        user_str = ''
        if 'edges' in sample.keys():
            adjacency_str = ''
            for u,v in sample['edges']:
                type_u = sample['rooms'][u]['room_type']
                type_v = sample['rooms'][v]['room_type']
                id_u = sample['rooms'][u]['id']
                id_v = sample['rooms'][v]['id']
                adjacency_str += f'({type_u}/"{id_u}", {type_v}/"{id_v}"), '
            adjacency_str = adjacency_str.strip(', ')
            if len(adjacency_str):
                user_str += f'adjacency constraints: {adjacency_str}. '
            del sample['edges']
        
        user_str += f"specifications: {str(sample['prompt'])}"
            
        if version == 'bd':
            instruction_str = 'you are to generate a floor plan in a JSON structure where each room is defined by polygon vertices, make sure to not overlap the polygons. you have to satisfy the adjacency constraints given as pairs of neighboring rooms; two connecting rooms, room1 and room2, are presented as (room1_type/"room1_id", room2_type/"room2_id"). you have to also match the specifications passed by the user in a JSON structure when they exist. when room area and total area requirements exist, make sure the polygon areas add up to the required number.'
            prompt_str = f"""<|start_header_id|>system<|end_header_id|> {instruction_str}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|> """

            prompt = tokenizer(f"{tokenizer.bos_token}{prompt_str}", add_special_tokens=False)
        
        else:
            instruction_str = 'you are to generate a floor plan in a JSON structure where each room is defined by polygon vertices, make sure to not overlap the polygons. you have to satisfy the requirements passed by the user in a JSON structure. when room area and total area requirements exist, make sure the polygon areas add up to the required number.'
            prompt_str = f"""<|start_header_id|>system<|end_header_id|> {instruction_str}<|eot_id|><|start_header_id|>user<|end_header_id|> {str(sample['prompt'])}<|eot_id|><|start_header_id|>assistant<|end_header_id|> """

            prompt = tokenizer(f"{tokenizer.bos_token}{prompt_str}", add_special_tokens=False)

        floorplan = tokenizer(f"\nOutput:\n{json.dumps({k: v for k, v in sample.items() if k != 'prompt'})}{tokenizer.eos_token}", add_special_tokens=False)

        input_ids = prompt['input_ids'] + floorplan['input_ids']
        attention_mask = [1] * (len(prompt['input_ids']) + len(floorplan['input_ids']))
        labels = [-100] * len(prompt['input_ids']) + floorplan['input_ids']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    if not testing:
        return dataset.map(
            process_sample, 
            remove_columns=list(dataset.features)
        )
    else:
        return dataset.map(
            lambda sample: {'input': json.dumps({k: v for k, v in sample.items() if k != 'prompt'}),
                            'prompt': f"{tokenizer.bos_token}Input:\n{str(sample['prompt'])}"}, 
            remove_columns=list(dataset.features)
        )
        # return dataset