# This software was developed by Ivi Chatzi, Nina Corvelo Benz, Eleni Straitouri, Stratis Tsirtsis, and Manuel Gomez Rodriguez.
# If you use this code, please cite the paper "Counterfactual Token Generation in Large Language Models" by the same authors.

from typing import List, Optional

import fire

import sys
import os
import json
import torch

# Add the src directory to the Python path
sys.path.append(os.path.abspath("src/llama3"))
from llama import Dialog, Llama
from llama.sampler import Sampler 

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    exp_name: str = "test",
    prior: bool = False,
    iteration: int = 1
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """

    with open(f"outputs/{exp_name}/intervention_{iteration}.json") as f:
        data = json.load(f)
        system = data["system"]
        query = data["query"]
        partial_response = data["response"]
        top_p = data["top_p"]
        top_k = data["top_k"]
        temperature = data["temperature"]
        start_from = data["start_from"]
        seed = data["seed"]
        sampler_type=data["sampler_type"]

    if top_p == "None":
        top_p = None
    if top_k == "None":
        top_k = None
    if sampler_type == "None":
        sampler_type = None
    
    with open(f"outputs/{exp_name}/rngstates_{iteration}.pt", "rb") as f:
        rngstates = torch.load(f)
    init_rng_state = rngstates[start_from,:]

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=seed,
        init_rng_state=init_rng_state,
        prior=prior
    )

    dialogs: List[Dialog] = [
        [   {"role": "system", "content": system},
            {"role": "user", "content": query}]
    ]

    if sampler_type is not None:
        if top_p is not None:
            sampler = Sampler(sampler_type=sampler_type,top_p=top_p)
        elif top_k is not None:
            sampler = Sampler(sampler_type=sampler_type,top_k=top_k)
    else:
        sampler = None

    results, rngstates = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        sampler=sampler,
        genstates=False,
        partial_response=partial_response,
    )

    output_dir = os.path.join("outputs", exp_name)

    output = {}
    output["system"] = system
    output["query"] = query
    output["response"] = results[0]["generation"]["content"]
    output["seed"] = seed
    output["temperature"] = temperature
    if top_p is not None:
        output["top_p"] = top_p
    else:
        output["top_p"] = "None"
    if top_k is not None:
        output["top_k"] = top_k
    else:
        output["top_k"] = "None"
    if sampler_type is not None:
        output["sampler_type"] = sampler_type
    else:
        output["sampler_type"] = "None"
    output["start_from"] = 0
    output["token_list"] = {ind: tok for ind, tok in enumerate(results[0]["generation"]["token_list"])}

    with open(os.path.join(output_dir, f"counterfactual_{iteration}.json"), "w") as f:
        json.dump(output, f, indent=4)

    with open(os.path.join(output_dir, f"intervention_{iteration+1}.json"), "w") as f:
        json.dump(output, f, indent=4)

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f">Counterfactual: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)

