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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))[:-4]

sys.path.append(os.path.abspath("src/llama3"))
from llama import Dialog, Llama
from llama.sampler import Sampler


# Llama paths
CKPT_DIR = f"{ROOT_DIR}/src/llama3/pretrained/Meta-Llama-3-8B-Instruct/"
TOKENIZER_PATH = f"{ROOT_DIR}/src/llama3/pretrained/Meta-Llama-3-8B-Instruct/tokenizer.model"

# Attributes to intervene on
INTERVENTION_KEY_MAP = {
    "Sex": "Sex",
    # "Name": "Name",
    "Race": "Race",
    "Ethnicity": "Eth",
}
INTERVENTION_KEY_OFFSET = {
    "Sex": 0,
    # "Name": 0,
    "Race": 0,
    "Ethnicity": 2,
}
# Intervention RNG
INTERV_RNG = np.random.default_rng(827649827698)

# Number of census data batches
N_CENSUS = 3

# Debug Flag
DEBUG = False

def save_interventions():
    census_df = pd.DataFrame()

    for i in range(1,4):
        with open(f"{ROOT_DIR}/outputs/census"+str(i)+"/factual.json", 'r') as file:
            data = json.load(file)
            response = data["response"].split("```")[1]
            
            census_df = pd.concat([census_df, pd.read_json(StringIO(response))], ignore_index=True)

    interventions = {}
    for column_name, col in census_df.items():
        if column_name != "Name":
            interventions[column_name]=col.unique().tolist()
            
    with open(f"{ROOT_DIR}/outputs/bias/census_interventions.json", "w") as outfile: 
        json.dump(interventions, outfile, indent=4)

    interventions_name={}
    for variable, intervention_values in interventions.items():
        if variable in ["Race","Ethnicity","Sex"]:
            interventions_name[variable]={}
            for i in intervention_values:
                interventions_name[variable][i] = census_df[census_df[variable]==i]["Name"].to_list()

    with open(f"{ROOT_DIR}/outputs/bias/census_name_interventions.json", "w") as outfile: 
        json.dump(interventions_name, outfile, indent=4)


def get_where_to_intervene_dict(token_list, intervention_keys):
    idx_dict={k:[] for k in intervention_keys}
    t=[]
    key=''

    for token_idx, token in token_list.items():
        if token in intervention_keys and key == '':
            t.append(token_idx)
            key=token
        elif token in {'\",\n',')\",\n'} and key!='':
            t.append(str(1+int(token_idx)))
            idx_dict[key].append(t)
            t=[]
            key=''

    return idx_dict

def parse_intervention_file(path):
    with open(path,'r') as f:
        intervention_dict = json.load(f)
    
    token_list = intervention_dict['token_list']

    factual_response_as_dict = json.loads(intervention_dict['response'].split('```')[1])

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

def main():
    # Parse and save all possible interventions per attribute
    save_interventions()
    interventions_dict_path = f"{ROOT_DIR}/outputs/bias/census_interventions.json"
    with open(interventions_dict_path, 'r') as f:
        interventions_dict = json.load(f)

    # Define sampler 
    sampler = Sampler(sampler_type='vocabulary')

    results = []
    for i in range(1,4):
        intervention_path = f"{ROOT_DIR}/outputs/census{i}/intervention_1.json"
        token_list, factual_response_as_dict, seed, temperature, system, query = parse_intervention_file(path=intervention_path)

        where_to_intervene_dict = get_where_to_intervene_dict(token_list, INTERVENTION_KEY_MAP.values())

        n_people = len(where_to_intervene_dict['Sex'])
        
        # Build llama
        generator = Llama.build(
            ckpt_dir=CKPT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            max_seq_len=8192,
            max_batch_size=4,
            seed=seed
        )
        with open(f"{ROOT_DIR}/outputs/census{i}/rngstates_1.pt", "rb") as f:
            rngstates = torch.load(f)

        dialogs: List[Dialog] = [
        [   {"role": "system", "content": system},
            {"role": "user", "content": query}]
        ] 


        for j in range(n_people):
            for key in INTERVENTION_KEY_MAP.keys():
                for intervention in interventions_dict[key]:
                    current_value = factual_response_as_dict[j][key]
                    if intervention == current_value:
                        continue

                    if DEBUG: 
                        print(intervention)
                
                    prefix_idx = where_to_intervene_dict[INTERVENTION_KEY_MAP[key]][j][0]
                    partial_response_prefix_str = ''.join(list(token_list.values())[:int(prefix_idx)+1+INTERVENTION_KEY_OFFSET[key]])
                    partial_response = partial_response_prefix_str + "\": \"" + intervention + "\",\n"

                    if DEBUG:
                        print(partial_response)

                    start_from  = int(where_to_intervene_dict[INTERVENTION_KEY_MAP[key]][j][1])
                        
                    if DEBUG:
                        print(start_from)

                    # set rng state for counterfactual after the intervention
                    init_rng_state = rngstates[start_from,:]
                    init_rng_state = torch.tensor(init_rng_state, device="cpu").to(torch.uint8)
                    generator.rng.set_state(init_rng_state)

                    # counterfactual generation
                    results_cf, rngstates_cf = generator.chat_completion(
                        dialogs,
                        max_gen_len=None,
                        temperature=temperature,
                        top_p=None,
                        top_k=None,
                        sampler=sampler,
                        genstates=False,
                        partial_response=partial_response,
                        auto_intervention=False
                    )

                    cf_response = results_cf[0]["generation"]["content"]
                    results.append((i, j, key, intervention, cf_response))
    
    df = pd.DataFrame(results, columns=['census_id', 'person_id', 'intervened_attribute', 'intervention', 'cf_response'])

    output_path = f'{ROOT_DIR}/outputs/bias/all_interventions_per_attribute_{i}.json'
    if not os.path.exists(f'{ROOT_DIR}/outputs/bias'):
        os.mkdir(f'{ROOT_DIR}/outputs/bias')
    
    df.to_json(output_path)


if __name__ == "__main__":
    fire.Fire(main)