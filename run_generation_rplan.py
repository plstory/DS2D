# from metrics import create_metrics, save_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from datasets import load_from_disk
from src.pred import extract_output_json
import torch
# from metrics.json_operations import save_metrics
# from pred import load_preds, get_dataset_without_prompt,load_preds_partial
# from metrics import create_metrics, create_prompt_metrics
# from metrics.calculate_prompt_metrics import calculate_prompt_metrics
# from utils import list_folders
import json
from tqdm import tqdm
from peft import PeftModel
import random
# import wandb
from copy import deepcopy
import sys
import argparse

def load_dataset(dataset_dir="datasets/rplan_converted", split="test", exprm=5):
    data_idx = exprm[0]
    dataset_name = f'{dataset_dir}/{data_idx}/'
    dataset = load_from_disk(dataset_name)

    loaded_dataset = dataset[split]
    return loaded_dataset

def load_model_z(model_id="meta-llama/Meta-Llama-3-8B-Instruct", exprm='all'
):
    model_id = model_id
    peft_model_id = "models/rplan_weights_variants/" + exprm
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    # model.load_adapter(peft_model_id)
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model = model.merge_and_unload()
    return model, tokenizer

def filter_key_in_list(dicts, filter_out='prompt'):
    return [{key: value for key, value in d.items() if key != filter_out} for d in dicts]


def predict_output_rplan(model, tokenizer, data, exprm, num_samples = 1, partial_prompt=0):

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
    pixel2len = 18/256
    pixel2area = pixel2len**2
    prompt = {}

    prompt_d = deepcopy(data)
    if partial_prompt == 0:
        prompt = deepcopy(data)
        for room_dict in prompt['rooms']:
            del room_dict['floor_polygon']
            for k in list(room_dict.keys()):
                if random.random() < 0.5:
                    del room_dict[k]
            if len(room_dict.keys()) == 0:
                del room_dict
        if len(prompt['rooms']) == 0:
            del prompt['rooms']
        rands = np.random.random(len(prompt.keys()))
        rands[np.argmax(rands)] = 1.0
        for idx, k in enumerate(list(prompt.keys())):
            if rands[idx] < 0.5:
                del prompt[k]
    if partial_prompt in [1,3]: # only_total_area
        prompt['total_area'] = prompt_d['total_area']
    if partial_prompt in [2,3]: # only_room_area
        rooms = prompt_d['rooms'].copy()
        if partial_prompt == 3:
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
    
    num_rooms = len(data['rooms'])
    
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

    if len(prompt.keys())>0:
        user_str += f'. additional constraints: {str(prompt)}'

    prompt_str = f"""<|start_header_id|>system<|end_header_id|> {instruction_str}<|eot_id|><|start_header_id|>user<|end_header_id|> {user_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|> """
    tokenizer.pad_token = tokenizer.eos_token
    
    model_input = tokenizer(f"{tokenizer.bos_token}{prompt_str}", add_special_tokens=False, return_tensors="pt").to("cuda")

    json_output = []

    with torch.no_grad():
        do_sample = num_samples>1
        # print('generating....')
        tmp = model.generate(**model_input, max_new_tokens=2800,do_sample=do_sample, num_return_sequences=num_samples, top_p = 0.8)
        # print('detaching....')
        outputs = tmp.detach().cpu()
        # print('writing....')
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            json_output.append(extract_output_json(decoded))
        del tmp, outputs, model_input
    return json_output, str(prompt), str(data)

def predict_outputs_multiple_rplan_v1(model, tokenizer, dataset, model_variant='TMP', num_samples = 1, results_dir = None, start_idx=0, end_idx=100):
    # results = []

    pp_dict = {
        0: 'full_prompt',
        1: 'only_total_area',
        2: 'only_room_area',
        3: 'some_room_area',
        4: 'only_bd'
    }
    if results_dir is None:
        results_dir = 'generations/rplan'
    
    # wandb.init(project="floorplans",name=f"generation: rplan {num_samples}: {start_idx}:{end_idx} {model_variant}")
    for idx ,example in enumerate(tqdm(dataset, desc="Predicting outputs")):
        
        for partial_prompt in [3,2,1,0]:
        # for partial_prompt in [4]:
            partial_prompt_method = pp_dict[partial_prompt]
            outputs, prompt, gt = predict_output_rplan(model, tokenizer, example, model_variant, num_samples, partial_prompt)
            out_dir = f'{results_dir}/{model_variant}/{start_idx+idx}/{partial_prompt_method}/'
            os.makedirs(os.path.dirname(out_dir), exist_ok=True)
            for f_idx, output in enumerate(outputs):
                # wandb.log({partial_prompt_method: len(str(output))}, step=start_idx+idx)
                fname = out_dir+f'{f_idx}.json'
                with open(fname,'w+') as f:
                    json.dump(output,f)
            # results.append(outputs)
            fname = out_dir + f'prompt.json'
            with open(fname,'w+') as f:
                f.write(prompt)
            fname = out_dir + f'ground_truth.json'
            with open(fname,'w+') as f:
                f.write(gt)
            del outputs, prompt, gt
    # return results

def main(args):
    num_samples = args.num_samples
    
    #parameter search for batch job
    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    exprm_search = ['5R','6R','7R','8R']
    if jobid is not None:
        jobid = int(jobid)
        exprm = exprm_search[jobid%4]
        if num_samples == 1:
            start_idx = 200 * (jobid//4)
            end_idx = start_idx + 200
        elif num_samples > 1:
            start_idx = 20 * (jobid//4)
            end_idx = start_idx + 20
    else:
        exprm = args.exprm

    print(f'exprm: {exprm}!!')
    print(f'exprm: {exprm}!!')
    print(f'exprm: {exprm}!!')
    print(f'exprm: {exprm}!!')
    model, tokenizer = load_model_z(exprm=exprm)
    model.eval()
    ds_dir = 'datasets/rplan'
    test_dataset = load_dataset(dataset_dir=ds_dir,exprm=exprm,split="train")
    np.random.seed(12345)

    idx_select = np.random.permutation(len(test_dataset))[start_idx:end_idx]
    test_dataset = test_dataset.select(idx_select)
    # x = filter_key_in_list(test_dataset)
    out_dir = 'generations/rplan'
    if num_samples == 1:
        out_dir+='_greedy'
    predict_outputs_multiple_rplan_v1(model, tokenizer, test_dataset, exprm, num_samples=num_samples, results_dir=out_dir,start_idx=start_idx,end_idx=end_idx)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exprm',type=str,help='model variant',default='5R')
    parser.add_argument('--num_samples',type=int,help='number of samples to generate',default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)