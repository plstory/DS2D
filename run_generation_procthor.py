from src.pred import predict_outputs, predict_outputs_multiple
from src.pred import load_model, load_dataset
import os
import numpy as np
import json
import sys
import argparse

def filter_key_in_list(dicts, filter_out='prompt'):
    return [{key: value for key, value in d.items() if key != filter_out} for d in dicts]

def main(args):
    
    jobid = os.getenv('SLURM_ARRAY_TASK_ID')
    num_samples = args.num_samples
    version = args.version
    exprm_search = ['full_prompt','mask','preset_mask']
    if jobid is not None:
        jobid = int(jobid)
        exprm = exprm_search[jobid%3]
        if num_samples == 1:
            start_idx = 200 * (jobid//3)
            end_idx = start_idx + 200
        elif num_samples > 1:
            start_idx = 20 * (jobid//3)
            end_idx = start_idx + 20
    else:
        start_idx = 0
        end_idx = 100
        if num_samples == 1:
            end_idx = 1000
        exprm = args.exprm

    print(f'exprm: {exprm}, num_samples: {num_samples}!!')
    print(f'exprm: {exprm}, num_samples: {num_samples}!!')
    print(f'exprm: {exprm}, num_samples: {num_samples}!!')
    print(f'exprm: {exprm}, num_samples: {num_samples}!!')

    if version == 'bd':
        model_dir = "models/procthor_weights_BD_variants/"
    else:
        model_dir = "models/procthor_weights_nonBD_variants/"

    model, tokenizer = load_model(model_dir=model_dir,exprm=exprm)
    #use validation  set here because test set was used for validation, just naming difference.
    test_dataset = load_dataset(dataset_name="datasets/procthor_converted",split="validation")
    np.random.seed(12345)
    idx_select = np.random.permutation(len(test_dataset))[start_idx:end_idx]
    test_dataset = test_dataset.select(idx_select)
    
    if num_samples > 1:
        result_dir = f'generations/procthor_{version}_sampling'
    else:
        result_dir = f'generations/procthor_{version}_greedy'

    predict_outputs_multiple(model, tokenizer, test_dataset, exprm, num_samples=num_samples,prompt_style={version}, result_dir=result_dir, start_idx=start_idx, end_idx=end_idx)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exprm',type=str,help='model variant',default='dropout')
    parser.add_argument('--num_samples',type=int,help='number of samples to generate',default=1)
    parser.add_argument('--version',type=str,help='version of procthor model is trained on, "bd" or "nonbd"',default='bd')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)