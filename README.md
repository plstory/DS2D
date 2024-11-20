# DStruct2Design: Data and Benchmarks for Data Structure Driven Generative Floor Plan Design

## Paper
Our paper is available [here](https://arxiv.org/abs/2407.15723)

### if you use this repository, please cite our work:
```
@misc{luo2024dstruct2designdatabenchmarksdata,
      title={DStruct2Design: Data and Benchmarks for Data Structure Driven Generative Floor Plan Design}, 
      author={Zhi Hao Luo and Luis Lara and Ge Ya Luo and Florian Golemo and Christopher Beckham and Christopher Pal},
      year={2024},
      eprint={2407.15723},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.15723}, 
}
```

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In our paper, we train a LLama3-8B-Instruct model. Training is enabled by [llama-receipe](https://github.com/meta-llama/llama-recipes/tree/main). You can either install llama-receipe, or install from `requirement.txt`

#### Install with llama-receipe:
```
pip install llama-recipes
```

#### Install from requirement.txt:
```
pip install -r requirements.txt
```

## Datasets

### ProcTHOR

You can download the converted ProcTHOR-10K dataset from [here](https://huggingface.co/datasets/ludolara/DStruct2Design) and put it under `datasets/procthor_converted/`

### RPLAN

RPLAN dataset needs to be requested from its [homepage](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/).

After it's obtained, save all the data (pngs) under `datasets/rplan/`. Then run our conversion script to convert it. The converted dataset will be saved under `datasets/rplan_converted/`:
```
python scripts/rplan_dataset_convert.py
```

## Pretrained Weights

The pretrained PEFT LoRA weights for all of our models can be obtained:

#### Weights for 4 model variants trained on RPLAN
```
https://drive.google.com/file/d/1cAYlEupNUGJefNdwkNaaq7fD3X3_P46D/view?usp=sharing
```

#### Weights for 3 bubble diagram model variants trained on ProcTHOR
```
https://drive.google.com/file/d/16cYPK6g_Ho4VbvjvBZIGHMzNTBWzcAZT/view?usp=drive_link
```


#### Weights for 3 constraint only (no bubble diagram) model variants trained on ProcTHOR
```
https://drive.google.com/file/d/13k-pBmhGhYthm4WbHzrRH7WjaSKNkTpq/view?usp=drive_link
```

After download, they can be un-compressed and put under their respective folder under `models/`.

## Training

Alternatively, these weights can be trained from scratch with the following command:

#### to train on ProcTHOR:

```
python -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --model_name meta-llama/Meta-Llama-3-8B-Instruct --batch_size_training 2 --num_epochs 8 --dataset "custom_dataset" --custom_dataset.file "procthor_dataset.py" --use_wandb False --wandb_config.project "floorplans" --output_dir procthor --exprm $EXPRM_VAR --load_peft False --ds_version $BD_VAR --load_peft False
```

here `$BD_VAR` and `$EXPRM_VAR` indicate the model variants to be trained as explained in Section 6.1 of our paper. 

`$BD_VAR` can be set to either `'bd'` or `'non_bd'`,

`$EXPRM_VAR` can be set to `'specific'`, `'mask'`, or `'preset_mask'`

#### to train on RPLAN:

```
python -m llama_recipes.finetuning --use_peft --peft_method lora --quantization --model_name meta-llama/Meta-Llama-3-8B-Instruct --batch_size_training 2 --num_epochs 5 --dataset "custom_dataset" --custom_dataset.file "rplan_dataset.py" --use_wandb False --wandb_config.project "floorplans" --output_dir rplan --exprm $EXPRM_VAR$ --load_peft False
```

for RPLAN, the model variant is decided by just `$EXPRM_VAR`.

`$EXPRM_VAR` can be 1 of `'5R'`, `'6R'`, `'7R'`, or `'8R'`. The differences between these variants are explained in Section 6.1 of our paper.

## Inference

To run genneration after the pretrained weights are obtained, do the following:

(note that you can run greedy or sample generations. In our experiments, we use both, and sampling is done with `num_samples` set to 20).

#### To run generation on PROCTHOR-trained models:

```
python run_generation_procthor.py --exprm $EXPRM_VAR --num_samples 1 --version $BD_VAR
```

`$BD_VAR` can be set to either `'bd'` or `'non_bd'`,

`$EXPRM_VAR` can be set to `'specific'`, `'mask'`, or `'preset_mask'`

It will load the trained model variant according to the variable.


#### To run generation on RPLAN-trained models:

```
python run_generation_rplan.py --exprm $EXPRM_VAR --num_samples 1
```

`$EXPRM_VAR` can be 1 of `'5R'`, `'6R'`, `'7R'`, or `'8R'`. 

It will load the trained model variant according to the variable.

## Evaluation

To evaluate generated results saved in `$RESULTS_DIR`, simply run the following command:

```
python run_metric.py $RESULTS_DIR
```

