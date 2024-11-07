# This software was developed by Ivi Chatzi, Nina Corvelo Benz, Eleni Straitouri, Stratis Tsirtsis, and Manuel Gomez Rodriguez.
# If you use this code, please cite the paper "Counterfactual Token Generation in Large Language Models" by the same authors.

from typing import List, Optional

import fire

import sys
import os
import json
import torch
from sampler import Sampler 

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

# Add the src directory to the Python path
sys.path.append(os.path.abspath("src/llama3"))
from llama import Dialog, Llama

def main(
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
        sampler_type = data["sampler_type"]
        model_family = data["model_family"]
        ckpt_dir = data["ckpt_dir"]
        tokenizer_path = data["tokenizer_path"]

    sampler = Sampler(sampler_type=sampler_type, top_p=top_p, top_k=top_k)
    
    with open(f"outputs/{exp_name}/rngstates_{iteration}.pt", "rb") as f:
        rngstates = torch.load(f)
    init_rng_state = rngstates[start_from,:]

    if model_family == "llama3":
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

        results, rngstates = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            sampler=sampler,
            genstates=False,
            partial_response=partial_response,
        )

    elif model_family == "mistral":

        # load the model
        mistral_tokenizer: MistralTokenizer = load_tokenizer(Path(ckpt_dir))
        tokenizer: Tokenizer = mistral_tokenizer.instruct_tokenizer.tokenizer

        model_cls = get_model_cls(ckpt_dir)
        model = model_cls.from_folder(Path(ckpt_dir), max_batch_size=max_batch_size, num_pipeline_ranks=1)
        
        messages: List[SystemMessage | UserMessage | AssistantMessage] = []
        messages += [SystemMessage(content=system)]
        messages += [UserMessage(content=query)]

        chat_completion_request = ChatCompletionRequest(messages=messages)
        tokenized = mistral_tokenizer.encode_chat_completion(chat_completion_request)
        tokens = tokenized.tokens

        # initialize a random number generator with the given seed
        rng = torch.Generator(device="cuda")
        rng.manual_seed(seed)

        if not prior:
            # turn init_rng_state into a torch.ByteTensor
            init_rng_state = torch.tensor(init_rng_state, device="cpu").to(torch.uint8)
            rng.set_state(init_rng_state)

        fixed_tokens = tokenizer.encode(partial_response, bos=False, eos=False)

        generated_tokens, _, _ = generate(  # type: ignore[operator]
            encoded_prompts = [tokens],
            model = model,
            max_tokens = max_seq_len,
            temperature = temperature,
            eos_id = tokenizer.eos_id,
            sampler = sampler,
            genstates = False,
            rng = rng,
            fixed_tokens=fixed_tokens
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


    output_dir = os.path.join("outputs", exp_name)

    output = {}
    output["system"] = system
    output["query"] = query
    output["response"] = results[0]["generation"]["content"]
    output["seed"] = seed
    output["temperature"] = temperature
    output["sampler_type"] = sampler_type
    output["top_p"] = top_p
    output["top_k"] = top_k
    output["model_family"] = model_family
    output["start_from"] = 0
    output["token_list"] = {ind: tok for ind, tok in enumerate(results[0]["generation"]["token_list"])}

    with open(os.path.join(output_dir, f"counterfactual_{iteration}.json"), "w") as f:
        json.dump(output, f, indent=4)

    with open(os.path.join(output_dir, f"intervention_{iteration+1}.json"), "w") as f:
        json.dump(output, f, indent=4)

    print("----------------")
    print("SYSTEM:", system)
    print("----------------")
    print("QUERY:", query)
    print("----------------")
    print("RESPONSE:", results[0]["generation"]["content"])
    print("----------------")


if __name__ == "__main__":
    fire.Fire(main)

