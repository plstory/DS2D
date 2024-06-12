import torch
from .extract_output_json import extract_output_json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import json
import os
import numpy as np
from copy import deepcopy
from peft import PeftModel
# import wandb

def update_progress_bar(bar):
    def update(_):
        bar.update()
    return update

def load_dataset(dataset_name="datasets/procthor_converted", split="test"):
    dataset = load_from_disk(dataset_name)
    loaded_dataset = dataset[split]
    return loaded_dataset

def load_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct", model_dir="models/procthor_weights_BD_variants/", exprm='full_prompt'
):
    model_id = model_id
    peft_model_id = os.path.join(model_dir,exprm)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    # model.load_adapter(peft_model_id)
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model = model.merge_and_unload()

    return model, tokenizer

def predict_output(model, tokenizer, data, num_samples = 1, prompt_style='old', partial_prompt = 0):
    datapoint = deepcopy(data)
    # 1: 'only_total_area',
    # 2: 'only_room_area',
    # 3: 'total_some_room'
    prompt = {}
    prompt['room_count'] = datapoint['room_count']
    if partial_prompt == 0:
        prompt = datapoint['prompt']
    if partial_prompt in [1,3]: # only_total_area
        prompt['total_area'] = datapoint['prompt']['total_area']
    if partial_prompt in [2,3]: # only_room_area
        rooms = datapoint['rooms'].copy()
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
    
    prompt = str(prompt)

    if prompt_style == 'old':
        eval_prompt = f"""Input:\n{prompt}\n Output:"""
    elif prompt_style == 'v7':
        instruction_str = 'you are to generate a floor plan in a JSON structure where each room is defined by polygon vertices, make sure to not overlap the polygons. you have to satisfy the requirements passed by the user in a JSON structure. when room area and total area requirements exist, make sure the polygon areas add up to the required number.'

        prompt_str = f"""<|start_header_id|>system<|end_header_id|> {instruction_str}<|eot_id|><|start_header_id|>user<|end_header_id|> {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|> """
    elif prompt_style == 'v8':
        instruction_str = 'you are to generate a floor plan in a JSON structure where each room is defined by polygon vertices, make sure to not overlap the polygons. you have to satisfy the adjacency constraints given as pairs of neighboring rooms; two connecting rooms, room1 and room2, are presented as (room1_type/"room1_id", room2_type/"room2_id"). you have to also match the specifications passed by the user in a JSON structure when they exist. when room area and total area requirements exist, make sure the polygon areas add up to the required number.'
        # eval_prompt = tokenizer(f"{tokenizer.bos_token}{prompt_str}", add_special_tokens=False)
        user_str = ''
        if 'edges' in datapoint.keys():
            adjacency_str = ''
            for u,v in datapoint['edges']:
                type_u = datapoint['rooms'][u]['room_type']
                type_v = datapoint['rooms'][v]['room_type']
                id_u = datapoint['rooms'][u]['id']
                id_v = datapoint['rooms'][v]['id']
                adjacency_str += f'({type_u}/"{id_u}", {type_v}/"{id_v}"), '
            adjacency_str = adjacency_str.strip(', ')
            if len(adjacency_str):
                user_str += f'adjacency constraints: {adjacency_str}. '
            del datapoint['edges']
        
        user_str += f"specifications: {prompt}"
        prompt_str = f"""<|start_header_id|>system<|end_header_id|> {instruction_str}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|> """
    
    tokenizer.pad_token = tokenizer.eos_token
    model_input = tokenizer(prompt_str, return_tensors="pt").to("cuda")
    model.eval()
    json_output = []
    del datapoint['prompt']
    ground_truth = str(datapoint)
    with torch.no_grad():
        do_sample = num_samples>1
        tmp = model.generate(**model_input, max_new_tokens=4000,do_sample=do_sample, num_return_sequences=num_samples, top_p = 0.8)
        outputs = tmp.detach().cpu()
        for output in outputs:
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            
            json_output.append(extract_output_json(decoded))
            del decoded
        del outputs, tmp
    del model_input
    return json_output, prompt, ground_truth

def predict_outputs_multiple(model, tokenizer, dataset, model_variant='TMP', num_samples = 1, prompt_style='bd', result_dir = None, start_idx=0, end_idx=100):
    # results = []
    pp_dict = {
        0: 'full_prompt',
        1: 'only_total_area',
        2: 'only_room_area',
        3: 'some_room_area'
    }
    if result_dir is None:
        if num_samples > 1:
            result_dir = f'generations/procthor_{prompt_style}_sampling'
        else:
            result_dir = f'generations/procthor_{prompt_style}_greedy'
    # wandb.init(project="floorplans",name=f"generation: proc{prompt_style} {num_samples}: {start_idx} + {model_variant}")
    for idx ,example in enumerate(tqdm(dataset, desc="Predicting outputs")):
        # for partial_prompt in [0]:
        for partial_prompt in [3,2,1,0]:
            outputs, prompt, gt = predict_output(model, tokenizer, example, num_samples,prompt_style,partial_prompt)

            partial_prompt_method = pp_dict[partial_prompt]
            out_dir = f'{result_dir}/{model_variant}/{start_idx+idx}/{partial_prompt_method}/'
            os.makedirs(os.path.dirname(out_dir), exist_ok=True)
            for f_idx, output in enumerate(outputs):
                # wandb.log({partial_prompt_method: len(str(output))}, step=start_idx+idx)
                fname = out_dir+f'{f_idx}.json'
                with open(fname,'w+') as f:
                    json.dump(output,f)
            fname = out_dir + f'prompt.json'
            with open(fname,'w+') as f:
                f.write(prompt)
            fname = out_dir + f'ground_truth.json'
            with open(fname,'w+') as f:
                f.write(gt)
            del outputs, prompt, gt
    # return results


def predict_outputs(model, tokenizer, dataset, model_variant='TMP', num_samples = 1):
    results = []
    for idx ,example in enumerate(tqdm(dataset, desc="Predicting outputs")):
        output = predict_output(model, tokenizer, example, num_samples)[0]

        fname = 'generations/{}/out_{}.json'.format(model_variant,idx)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname,'w+') as f:
            json.dump(output,f)
        results.append(output)
    return results

def save_preds(results, output_file="result/output.json"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

# def load_preds(output_file="result/output.json"):
#     with open(output_file, 'r') as file:
#         return json.load(file)

def load_preds_partial(directory="generations/full_prompt"):
    all_data = []
    files_with_ids = [f for f in os.listdir(directory) if (f.endswith(".json") and not f.startswith('prompt'))]
    # Sort files based on numeric value in the filename
    files_with_ids.sort(key=lambda x: int(x.split('.')[0]))

    for filename in files_with_ids:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            all_data.append(data)

    return all_data

def get_dataset_without_prompt(dicts, filter_out='prompt'):
    return [{key: value for key, value in d.items() if key != filter_out} for d in dicts]

def get_only_prompt_dataset(dicts, filter_out='prompt'):
    return [d[filter_out] for d in dicts if filter_out in d]
