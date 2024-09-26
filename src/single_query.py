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
    top_p: float = 0.9,
    top_k: int = 10,
    sampler_type: str = 'vocabulary',
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    seed: int = 42,
    query: str = "give me a recipe for moussaka",
    exp_name: str = "test",
    system: str = "Keep your replies short and to the point but don't give single word answers.",
    genstates: bool = False
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        seed=seed
    )

    dialogs: List[Dialog] = [
        [   {"role": "system", "content": system},
            {"role": "user", "content": query}]
    ]


    sampler = Sampler(sampler_type=sampler_type, top_p=top_p, top_k=top_k)

    results, rngstates = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        sampler=sampler,
        genstates=genstates
    )

    output_dir = os.path.join("outputs", exp_name)
    # if a folder with the experiment name exists, delete its contents, otherwise create it
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    else:
        os.makedirs(output_dir)

    if genstates:
        rngstates = torch.tensor(rngstates, dtype=torch.uint8)
        # save the rngstates tensor
        torch.save(rngstates, os.path.join(output_dir, 'rngstates_1.pt'))

    output = {}
    output["system"] = system
    output["query"] = query
    output["response"] = results[0]["generation"]["content"]
    output["seed"] = seed
    output["temperature"] = temperature
    if sampler_type is not None:
        output["sampler_type"] = sampler_type
    else:
        output["sampler_type"] = "None"
    if top_p is not None:
        output["top_p"] = top_p
    else:
        output["top_p"] = "None" 
    if top_k is not None:
        output["top_k"] = top_k
    else:
        output["top_k"] = "None"
        
    # save the output of the factual generation
    with open(os.path.join(output_dir, "factual.json"), "w") as f:
        json.dump(output, f, indent=4)

    # prepare the intervention file
    output["start_from"] = 0
    output["token_list"] = {ind: tok for ind, tok in enumerate(results[0]["generation"]["token_list"])}
    with open(os.path.join(output_dir, "intervention_1.json"), "w") as f:
        json.dump(output, f, indent=4)

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)

