# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse

from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux
from train_utils import train_and_evaluate
import pandas as pd
from datasets import Dataset 

def run(args):
    #### Prepare datasets Prepare data for training
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            model_inputs = tokenizer(['predict: ' + text for text in examples['question']], max_length=args.max_input_length, truncation=True)
            expl_model_inputs = tokenizer(['explain: ' + text for text in examples['question']], max_length=args.max_input_length, truncation=True)
            model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
            model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['result'], max_length=256, truncation=True)
                rationale_output_encodings = tokenizer(examples['answer'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']
            model_inputs['aux_labels'] = rationale_output_encodings['input_ids']

            return model_inputs

    train = pd.read_csv("gsm8k.csv", )
    # split the data into train and valid
    train = train.sample(frac=1)
    train = train.reset_index(drop=True)
    train = train[:int(0.8*len(train))]
    valid = train[int(0.8*len(train)):]
    
    datasets = DatasetDict({
        'train': Dataset.from_pandas(train),
        'valid': Dataset.from_pandas(valid),
    })

    tokenized_datasets = datasets.map(
        tokenize_function,
        remove_columns=['question', 'answer', 'result'],
        batched=True
    )

    if args.model_type == 'standard':
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text(tokenizer)
        else:
            compute_metrics = compute_metrics_equation(tokenizer)


    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--type_rationale', type=str, default='if_else')
    parser.add_argument('--data_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="/kaggle/input/deepseek-math")

    args = parser.parse_args()

    # dic = {
    #     'dataset': 'esnli',
    #     'subsample': 1.0,
    #     'alpha': 0.5,
    #     'max_steps': 10000,
    #     'eval_steps': 1,
    #     'batch_size': 2,
    #     'optimizer_name': 'AdamW',
    #     'lr': 5e-05,
    #     'run': 0,
    #     'from_pretrained': 'google/t5-v1_1-base',
    #     'label_type': 'gt',
    #     'llm': 'palm',
    #     'max_input_length': 1024,
    #     'grad_steps': 1,
    #     'local_rank': -1,
    #     'gen_max_len': 64,
    #     'parallelize': False,
    #     'model_type': 'task_prefix',
    #     'bf16': False,
    #     'no_log': False,
    #     'output_rationale': False,
    #     'type_rationale': 'paper',
    #     'data_size': 1
    # }
    # from types import SimpleNamespace
    # args = SimpleNamespace(**dic)

    run(args)
    