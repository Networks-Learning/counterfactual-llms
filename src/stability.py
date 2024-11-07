# This software was developed by Ivi Chatzi, Nina Corvelo Benz, Eleni Straitouri, Stratis Tsirtsis, and Manuel Gomez Rodriguez.
# If you use this code, please cite the paper "Counterfactual Token Generation in Large Language Models" by the same authors.

from typing import List, Optional
import fire

import sys
import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm 
import warnings
from sampler import Sampler

# mistral stuff
from mistral_common.tokens.tokenizers.base import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    UserMessage,
    SystemMessage,
    BaseMessage
)
from mistral_inference.generate import generate
# Add the src directory to the Python path
sys.path.append(os.path.abspath("src/mistral-inference/src"))
from mistral_inference.main import load_tokenizer, get_model_cls
from pathlib import Path

warnings.filterwarnings("ignore")
tqdm.pandas()
# Add the src directory to the Python path
sys.path.append(os.path.abspath("src/llama3"))
from llama import Dialog, Llama
DEBUG = False

def stability_query(
    generator,
    temperature: float = 0.6,
    top_p: float = None,
    top_k: int = None,
    sampler=None,
    max_gen_len: Optional[int] = None,
    max_seq_len: int = 512,
    seed: int = 42,
    intervention_position_seed: int = 42,
    num_interventions: int = 1,
    query: str = "give me a recipe for moussaka",
    system: str = "Keep your replies short and to the point but don't give single word answers.",
    tokenizer: Tokenizer = None,
    mistral_tokenizer: MistralTokenizer = None,
    rng: torch.Generator = None,
    model_family: str = 'llama3'
    ):
    """
    Generates factual, counterfactual and response from prior for a given query.
    Intervention is automatically done on a random token.

    Args:
        generator: already built model
        num_interventions: number of automatic interventions for each query
    """

    if model_family == 'llama3':
        # reset model rng to seed
        generator.rng.manual_seed(seed)

        dialogs: List[Dialog] = [
            [   {"role": "system", "content": system},
                {"role": "user", "content": query}]
        ]
        # generate factual response
        results, rngstates = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            sampler=sampler,
            genstates=True
        )
    elif model_family == 'mistral':
        # initialize the random number generator with the given seed
        rng.manual_seed(seed)

        dialogs: List[SystemMessage | UserMessage | AssistantMessage] = []
        dialogs += [SystemMessage(content=system)]
        dialogs += [UserMessage(content=query)]

        chat_completion_request = ChatCompletionRequest(messages=dialogs)
        tokenized = mistral_tokenizer.encode_chat_completion(chat_completion_request)
        tokens = tokenized.tokens

        generated_tokens, _, rngstates = generate(  # type: ignore[operator]
            encoded_prompts = [tokens],
            model = generator,
            max_tokens = max_seq_len,
            temperature = temperature,
            eos_id = tokenizer.eos_id,
            sampler = sampler,
            genstates = True,
            rng = rng
        )

        results = [
            {
                "generation": {
                    "role": "assistant",
                    "content": tokenizer.decode(t),
                    "token_list": [tokenizer.decode([x]) for x in t],
                },
            }
            for t in generated_tokens
        ]
        
    output = {}
    output["factual_response"] = results[0]["generation"]["content"]
    output["temperature"] = temperature
    if top_p is not None:
        output["sampler_param"] = top_p
    elif top_k is not None:
        output["sampler_param"] = top_k
    else:
        output["sampler_param"] = 0

    if DEBUG:
        if model_family == 'mistral':
            dialogs = [dialogs]
        for dialog, result in list(zip(dialogs, results)):
            for msg in dialog:
                if model_family == 'llama3':
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                elif model_family == 'mistral':
                    print(f"{msg.role.capitalize()}: {msg.content}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")
    

    if model_family == 'llama3':
        rngstates_factual = torch.tensor(rngstates, dtype=torch.uint8)
    elif model_family == 'mistral':
        rngstates_factual = torch.tensor(rngstates, dtype=torch.uint8, device=generator.device)

    output["token_list"] = {ind: tok for ind, tok in enumerate(results[0]["generation"]["token_list"])}

    # initialise rng for intervention positions
    interv_rng = torch.Generator(device="cuda")
    interv_rng.manual_seed(intervention_position_seed)

    num_tokens=len(output["token_list"])
    output["cf_response"] = []
    output["prior_response"] = []
    output["interventions"] = []
    
    num_interventions = min(num_interventions, num_tokens)
    for i in range(num_interventions):
        # randomly select intervention position
    
        low = max(num_tokens*i // num_interventions, 1)
        high = min(num_tokens*(i+1) // num_interventions, num_tokens-2)
        start_from = torch.randint(low,high,(1,),generator=interv_rng, device=interv_rng.device) if low < high else i+1
        output["interventions"].append(int(start_from))

        partial_tokens = [output["token_list"][i] for i in range(start_from)]
        partial_response = ''.join(partial_tokens)

        # set rng state for counterfactual after the intervention
        init_rng_state = rngstates_factual[start_from,:]
        init_rng_state = torch.tensor(init_rng_state, device="cpu").to(torch.uint8)
        if model_family == 'llama3':
            generator.rng.set_state(init_rng_state)

            # counterfactual generation
            results_cf, rngstates_cf = generator.chat_completion(
                dialogs,
                max_gen_len=max_gen_len,
                temperature=temperature,
                sampler=sampler,
                genstates=False,
                partial_response=partial_response,
                auto_intervention=True
            )

        elif model_family == 'mistral':
            rng.set_state(init_rng_state)

            fixed_tokens = tokenizer.encode(partial_response, bos=False, eos=False)

            generated_tokens, _, _ = generate(  # type: ignore[operator]
                encoded_prompts = [tokens],
                model = generator,
                max_tokens = max_seq_len,
                temperature = temperature,
                eos_id = tokenizer.eos_id,
                sampler = sampler,
                genstates = False,
                rng = rng,
                fixed_tokens=fixed_tokens,
                auto_intervention=True
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
            


        output["cf_response"].append(results_cf[0]["generation"]["content"])
        if DEBUG:
            # if model_family == 'mistral/':
                # dialogs = [dialogs]        
            for dialog, result in zip(dialogs, results_cf):
                for msg in dialog:
                    if model_family == 'llama3':
                        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                    elif model_family == 'mistral':
                        print(f"{msg.role.capitalize()}: {msg.content}\n")
                        
                print(
                    f">Counterfactual: {result['generation']['content']}"
                )
                print("\n==================================\n")

        if model_family == 'llama3':
            # reset model rng to seed for prior
            generator.rng.manual_seed(seed)

            results_prior, rngstates_prior = generator.chat_completion(
                dialogs,
                max_gen_len=max_gen_len,
                temperature=temperature,
                sampler=sampler,
                genstates=False,
                partial_response=partial_response,
                auto_intervention=True
            )
        elif model_family == 'mistral':
            # reset model rng to seed for prior
            rng.manual_seed(seed)

            generated_tokens, _, _ = generate(  # type: ignore[operator]
                encoded_prompts=[tokens],
                model=generator,
                max_tokens=max_seq_len,
                temperature=temperature,
                eos_id=tokenizer.eos_id,
                sampler=sampler,
                genstates=False,
                rng=rng,
                fixed_tokens=fixed_tokens,
                auto_intervention=True
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


        output["prior_response"].append(results_prior[0]["generation"]["content"])

        if DEBUG:
            for dialog, result in zip(dialogs, results_prior):
                for msg in dialog:
                    if model_family == 'llama3':
                        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                    elif model_family == 'mistral':
                        print(f"{msg.role.capitalize()}: {msg.content}\n")
                print(
                    f">Prior: {result['generation']['content']}"
                )
                print("\n==================================\n")
    return output

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = None,
    top_k: int = None,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    seed: int = 42,
    num_interventions: int = 1,
    output_dir: str = None,
    output_file_params: str = None,
    output_file_responses: str = None,
    input_file: str = None,
    system: str = "Keep your replies short and to the point.",
    chunk_size: int = 10,
    top_p_intervention: float = 0.9,
    intervention_seed: int = 42,
    intervention_position_seed: int = 42,
    model_family: str = 'llama3',
    categorical: bool = False):
    """
    Builds model, reads input file with queries, calls stability_query twice per query, writes output+params.
    stability_query is called with top-p token and top-p position samplers.

    Args:
        num_interventions: number of random interventions per query
        chunk_size: number of queries per input chunk (input is processed in chunks and output is written every chunk)
    """
    
    if model_family == 'llama3':
        # build llama
        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            seed=seed
        )
        # Dummy variables for compatibility with mistral
        tokenizer = None
        mistral_tokenizer = None
        rng = None

    elif model_family == 'mistral':
        # load the model 
        mistral_tokenizer: MistralTokenizer = load_tokenizer(Path(ckpt_dir))
        tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

        model_cls = get_model_cls(ckpt_dir)
        generator = model_cls.from_folder(Path(ckpt_dir), max_batch_size=max_batch_size, num_pipeline_ranks=1)

        # initialize a random number generator with the given seed
        rng = torch.Generator(device="cuda")
     
    # Initialize samplers
    samplers = dict()
    for pp in top_p:
        top_p_token = Sampler(sampler_type="top-p token", 
                                  top_p=pp,
                                  top_p_intervention=top_p_intervention,
                                  intervention_seed=intervention_seed)
        samplers['top-p token '+str(pp)]=top_p_token
    
    for kk in top_k:
        top_k_token = Sampler(sampler_type="top-k token",
                          top_k=kk,
                          top_p_intervention=top_p_intervention,
                          intervention_seed=intervention_seed)
        samplers['top-k token '+str(kk)] = top_k_token
    
    vocabulary = Sampler(sampler_type='vocabulary',
                         top_p_intervention=top_p_intervention,
                         intervention_seed=intervention_seed)
    samplers['vocabulary'] = vocabulary

    if categorical:
        samplers['categorical'] = Sampler(sampler_type='categorical',
                                          top_p_intervention=top_p_intervention,
                                          intervention_seed=intervention_seed)

    df = pd.read_parquet(input_file)
    if DEBUG:
        df=df.head(n=1)


    def stability(row, sampler,temperature):
        output = stability_query(
            temperature=temperature,
            top_p=sampler.top_p,
            top_k=sampler.top_k,
            sampler=sampler,
            max_seq_len=max_seq_len,
            max_gen_len=max_gen_len,
            seed=seed,
            intervention_position_seed=intervention_position_seed,
            num_interventions=num_interventions,
            query=row['question'],
            system=system,
            generator=generator,
            tokenizer=tokenizer,
            mistral_tokenizer=mistral_tokenizer,
            rng=rng, 
            model_family=model_family,
        )
        return output['factual_response'], output['cf_response'], output['prior_response'], output['interventions'], output['token_list'], sampler.sampler_type, output['sampler_param'], output['temperature']
    
    file_path = Path(f"{output_dir}/{output_file_responses}.parquet")
    if file_path.exists():
        os.remove(file_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # split input into chunks, call stability_query for queries in chunk and write chunk output to file
    for df_part in tqdm(np.array_split(df, max(1,len(df)//chunk_size))):
        # run stability for all sampler types except vocabulary, with temperature=0.6
        for sampler_type in tqdm(samplers.keys(), leave=False):
            if sampler_type=='vocabulary':
                continue
            if DEBUG:
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print(sampler_type)
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            
            df_part[['factual_response',
            'cf_response',
            'prior_response',
            'interventions',
            'token_list',
            'sampler_type',
            'sampler_param',
            'temperature']]=df_part.progress_apply(
                stability, 
                sampler=samplers[sampler_type],
                temperature=0.6, 
                axis=1, 
                result_type='expand')
            if file_path.exists():
                df_part.to_parquet(file_path, engine='fastparquet', append=True)
            else:
                df_part.to_parquet(file_path, engine='fastparquet')
    

    # run stability for vocabulary or categorical sampler, all temperatures
    for df_part in tqdm(np.array_split(df, max(1,len(df)//chunk_size))):
        for t in tqdm(temperature, leave=False):
            if DEBUG:
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print(t)
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            if categorical:
                sampler=samplers['categorical']
            else:
                sampler=samplers['vocabulary']
            
            df_part[['factual_response',
            'cf_response',
            'prior_response',
            'interventions',
            'token_list',
            'sampler_type',
            'sampler_param',
            'temperature']]=df_part.progress_apply(
                stability, 
                sampler=sampler,
                temperature=t, 
                axis=1, 
                result_type='expand')
            if file_path.exists():
                df_part.to_parquet(file_path, engine='fastparquet', append=True)
            else:
                df_part.to_parquet(file_path, engine='fastparquet')


    # save params to file
    param_file_path=Path(f"{output_dir}/{output_file_params}.json")
    if param_file_path.exists():
        os.remove(param_file_path)
    params={"seed": seed,
            "top_p": top_p,
            "top_k": top_k,
            "top_p_intervention": top_p_intervention,
            "interevention_seed": intervention_seed,
            "intervention_position_seed": intervention_position_seed,
            "system":system,
            "temperature": temperature,
            "input_file": input_file,
            "output_dir":output_dir}
    print(params)
    with open(param_file_path, 'w') as fp:
        json.dump(params, fp)

if __name__ == "__main__":
    fire.Fire(main)