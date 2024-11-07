# This software was developed by Ivi Chatzi, Nina Corvelo Benz, Eleni Straitouri, Stratis Tsirtsis, and Manuel Gomez Rodriguez.
# If you use this code, please cite the paper "Counterfactual Token Generation in Large Language Models" by the same authors.

from typing import List
import fire
import os
import json
import sys 
import numpy as np
import torch
import pandas as pd 
from io import StringIO
from sampler import Sampler

# ROOT_DIR = '.'

# mistral stuff
from mistral_common.tokens.tokenizers.base import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
    SystemMessage
)
from mistral_inference.generate import generate
# Add the src directory to the Python path
sys.path.append(os.path.abspath("src/mistral-inference/src"))
from mistral_inference.main import load_tokenizer, get_model_cls
from pathlib import Path


sys.path.append(os.path.abspath("src/llama3"))
from llama import Dialog, Llama




# Attributes to intervene on
INTERVENTION_KEY_MAP = {
    "Sex": "Sex",
    # "Name": "Name",
    "Race": "Race",
    # "Ethnicity": "Eth",
}
INTERVENTION_KEY_OFFSET = {
    "Sex": 0,
    # "Name": 0,
    "Race": 0,
    # "Ethnicity": 2,
}
# START_TOKENS = {
#     'llama3': { 
#         "Income", 
#         "Occup"
#         },
#     'mistral': {
#         "Income",
#         "Occ"
#     }
# }
START_TOKENS = {
    'llama3': {
        "Sex": "Income",
        "Race": "Occup" 
    },
    'mistral':{
        "Sex": "Income",
        "Race": "Occ"
    }
}
EOATTR_TOKEN = {
    'llama3': {
        '\",\n',
        ')\",\n',
        '\",'
    },
    'mistral': {
        '\",',
        ')\",'
    }


}
# Intervention RNG
INTERV_RNG = np.random.default_rng(827649827698)

# Number of census data batches
N_CENSUS = 3

# Debug Flag
DEBUG = False

def fix_income(value):
    if str(value)[0]=='$':
        value = int(''.join(value.split(','))[1:])
    return value

def save_interventions(model_family='llama3'):
    census_df = pd.DataFrame()

    for i in range(1,4):
    
        with open(f"outputs/{model_family}/census"+str(i)+"/factual.json", 'r') as file:
            data = json.load(file)
            response = data["response"].split("```")[1]
            if model_family == 'mistral':
                response = response.split('json')[1]
            elif model_family == 'llama3':
                if i == 1:
                    response = response.split('json')[1]
            
            census_df = pd.concat([census_df, pd.read_json(StringIO(response))], ignore_index=True)
    
    interventions = {}
    for column_name, col in census_df.items():
        if column_name != "Name":
            interventions[column_name] = col.unique().tolist()
            
    path = f"outputs/{model_family}/bias"
    if not os.path.exists(path):
        os.mkdir(path)

    with open(f"{path}/census_interventions.json", "w") as outfile: 
        json.dump(interventions, outfile, indent=4)


def get_where_to_intervene_dict(token_list, intervention_keys, model_family):
    idx_dict={k:[] for k in intervention_keys}
    t=[]
    key=''

    for token_idx, token in token_list.items():
        if token in intervention_keys and key == '':
            t.append(token_idx)
            key=token
        elif token in EOATTR_TOKEN[model_family] and key!='':
            t.append(str(1+int(token_idx)))
            idx_dict[key].append(t)
            t=[]
            key=''

    return idx_dict

def get_where_to_intervene_dict_direct(token_list, intervention_keys, model_family):
    idx_dict={k:[] for k in intervention_keys}
    # keys_found=set()
    key=''

    for token_idx, token in token_list.items():
        if token in intervention_keys: # and not (token in keys_found):
            idx_dict[token].append([token_idx])
            # keys_found.add(token)
            # if len(keys_found)==len(intervention_keys):
                # keys_found=set()
        elif token == START_TOKENS[model_family][intervention_keys[0]] and key=='':
            key=token
        elif key!='' and token=="\":" :
            idx_dict[intervention_keys[0]][-1].append(token_idx)
            key=''

    return idx_dict

def parse_intervention_file(path, census_no, model_family):
    with open(path,'r') as f:
        intervention_dict = json.load(f)
    
    token_list = intervention_dict['token_list']

    if model_family == 'llama3':
        response = intervention_dict['response'].split('```')[1]
        if census_no == 1:
            response = response.split('json')[1]
        factual_response_as_dict = json.loads(response)
    elif model_family == 'mistral':
        response = intervention_dict['response'].split('```')[1].split('json')[1]
        factual_response_as_dict = json.loads(response)

    seed = intervention_dict['seed']
    temperature = intervention_dict['temperature']
    system = intervention_dict['system']
    query = intervention_dict['query']
    return token_list, factual_response_as_dict, seed, temperature, system, query
     
def do_intervention(person_idx, list_of_people, interventions_dict, intervention_key):

    current_value = list_of_people[person_idx][intervention_key]
    
    if DEBUG:
        print(current_value)

    all_possible_intervention_values = interventions_dict[intervention_key]

    if DEBUG:
        print(all_possible_intervention_values)

    interventions = INTERV_RNG.choice(all_possible_intervention_values, 2, replace=False, shuffle=False)

    for inter in interventions:
        if current_value != inter:
            return inter

def main(i=1, direct=False, model_family='llama3', prior=True, attribute_to_intervene='Sex'):
    # Parse and save all possible interventions per attribute
    save_interventions(model_family)

    if model_family == 'llama3':
        CKPT_DIR = "src/llama3/pretrained/Meta-Llama-3-8B-Instruct/"
        TOKENIZER_PATH = "src/llama3/pretrained/Meta-Llama-3-8B-Instruct/tokenizer.model"
    elif model_family == 'mistral':
        CKPT_DIR = "src/mistral-inference/8B-Instruct/"
        TOKENIZER_PATH = "src/mistral-inference/8B-Instruct/"



    interventions_dict_path = f"outputs/{model_family}/bias/census_interventions.json"
    with open(interventions_dict_path, 'r') as f:
        interventions_dict = json.load(f)

    # Define sampler 
    sampler = Sampler(sampler_type='vocabulary')

    if model_family == 'llama3':
        # Build llama
        generator = Llama.build(
            ckpt_dir=CKPT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            max_seq_len=8192,
            max_batch_size=4,
            seed=42
        )
    
    elif model_family == 'mistral':
        # load the model
        mistral_tokenizer: MistralTokenizer = load_tokenizer(Path(TOKENIZER_PATH))
        tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

        model_cls = get_model_cls(CKPT_DIR)
        generator = model_cls.from_folder(Path(CKPT_DIR), max_batch_size=4, num_pipeline_ranks=1)
        
        rng = torch.Generator(device="cuda")

    # for i in range(1,4):
    results = []
    intervention_path = f"outputs/{model_family}/census{i}/intervention_1.json"
    token_list, factual_response_as_dict, seed, temperature, system, query = parse_intervention_file(path=intervention_path, census_no=i, model_family=model_family)
    
    if model_family == 'llama3':
        generator.rng.manual_seed(seed)
    elif model_family == 'mistral':
        rng.manual_seed(seed)

    where_to_intervene_dict = get_where_to_intervene_dict(token_list, [attribute_to_intervene], model_family)
    if direct:
        where_to_intervene_dict_direct = get_where_to_intervene_dict_direct(token_list, [attribute_to_intervene], model_family)
    
    n_people = len(where_to_intervene_dict[attribute_to_intervene])
    
    with open(f"outputs/{model_family}/census{i}/rngstates_1.pt", "rb") as f:
        rngstates = torch.load(f)

    if model_family == 'llama3':
        dialogs: List[Dialog] = [
        [   {"role": "system", "content": system},
            {"role": "user", "content": query}]
        ] 
    elif model_family == 'mistral':
        dialogs: List[SystemMessage | UserMessage | AssistantMessage] = []
        dialogs += [SystemMessage(content=system)]
        dialogs += [UserMessage(content=query)]

        chat_completion_request = ChatCompletionRequest(messages=dialogs)
        tokenized = mistral_tokenizer.encode_chat_completion(chat_completion_request)
        tokens = tokenized.tokens
    
    for j in range(n_people):
        for key in [attribute_to_intervene]:
            for intervention in interventions_dict[key]:
                current_value = factual_response_as_dict[j][key]
                if intervention == current_value:
                    continue

                # if DEBUG: 
                #     print(intervention)
            
                prefix_idx = where_to_intervene_dict[INTERVENTION_KEY_MAP[key]][j][0]
                partial_response_prefix_str = ''.join(list(token_list.values())[:int(prefix_idx)+1+INTERVENTION_KEY_OFFSET[key]])
                partial_response = partial_response_prefix_str + "\": \"" + intervention + "\",\n"

                if direct:
                    intervention_idx = where_to_intervene_dict[INTERVENTION_KEY_MAP[key]][j][1]
                    start_from = int(where_to_intervene_dict_direct[INTERVENTION_KEY_MAP[key]][j][1])
                    partial_response = partial_response + ''.join(list(token_list.values())[int(intervention_idx):start_from]) +"\":"
                    start_from += 1
                else:
                    start_from = int(where_to_intervene_dict[INTERVENTION_KEY_MAP[key]][j][1])

                if DEBUG:
                    print(partial_response)

                
                    
                # if DEBUG:
                #     print(start_from)

                # set rng state for counterfactual after the intervention
                init_rng_state = rngstates[start_from,:]
                init_rng_state = torch.tensor(init_rng_state, device="cpu").to(torch.uint8)
                
                if model_family == 'llama3':
                    generator.rng.set_state(init_rng_state)

                    # counterfactual generation
                    results_cf, rngstates_cf = generator.chat_completion(
                        dialogs,
                        max_gen_len=None,
                        temperature=temperature,
                        sampler=sampler,
                        genstates=False,
                        partial_response=partial_response,
                        auto_intervention=False
                    )
                elif model_family == 'mistral':
                    rng.set_state(init_rng_state)

                    fixed_tokens = tokenizer.encode(partial_response, bos=False, eos=False)

                    
                    generated_tokens, _, _ = generate(  # type: ignore[operator]
                        encoded_prompts = [tokens],
                        model = generator,
                        max_tokens = 8192,
                        temperature = temperature,
                        eos_id = tokenizer.eos_id,
                        sampler = sampler,
                        genstates = False,
                        rng = rng,
                        fixed_tokens=fixed_tokens,
                        auto_intervention=False
                    )

                    results_cf = [
                        {
                            "generation": {
                                "role": "assistant",
                                "content": tokenizer.decode(t),
                                "token_list": [tokenizer.decode([x]) for x in t],
                            },
                        }
                        for t in generated_tokens
                    ]

                cf_response = results_cf[0]["generation"]["content"]

                if prior:
                    if model_family == 'llama3':
                        generator.rng.manual_seed(seed)

                        results_prior, rngstates_prior = generator.chat_completion(
                            dialogs,
                            max_gen_len=None,
                            temperature=temperature,
                            sampler=sampler,
                            genstates=False,
                            partial_response=partial_response,
                            auto_intervention=False
                        )
                    elif model_family == 'mistral':
                        rng.manual_seed(seed)
                        
                        generated_tokens, _, _ = generate(  # type: ignore[operator]
                            encoded_prompts=[tokens],
                            model=generator,
                            max_tokens=8192,
                            temperature=temperature,
                            eos_id=tokenizer.eos_id,
                            sampler=sampler,
                            genstates=False,
                            rng=rng,
                            fixed_tokens=fixed_tokens,
                            auto_intervention=False
                        )
                        results_prior = [
                            {
                                "generation": {
                                    "role": "assistant",
                                    "content": tokenizer.decode(t),
                                    "token_list": [tokenizer.decode([x]) for x in t],
                                },
                            }
                            for t in generated_tokens
                        ]

                    prior_response = results_prior[0]["generation"]["content"]
                    results.append((i, j, key, intervention, cf_response, prior_response))
                else: 
                    results.append((i, j, key, intervention, cf_response))
                if DEBUG:
                    print("\nCounterfactual:")
                    print(cf_response)
                    if prior:
                        print("\nPrior:")
                        print(prior_response)
             

    columns = ['census_id', 'person_id', 'intervened_attribute', 'intervention', 'cf_response']
    if prior:
        columns.append('prior_response')
    df = pd.DataFrame(results, columns=columns)   

    if direct:
        output_path= f'outputs/{model_family}/bias/direct_interventions_{attribute_to_intervene}_{i}.json'
    else:
        output_path = f'outputs/{model_family}/bias/total_interventions_{attribute_to_intervene}_{i}.json'
    print("Saving output at "+output_path)

    if not os.path.exists(f'outputs/{model_family}/bias'):
        os.mkdir(f'outputs/{model_family}/bias')
    
    df.to_json(output_path)
    
if __name__ == "__main__":
    fire.Fire(main)