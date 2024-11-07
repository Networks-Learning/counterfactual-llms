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
    genstates: bool = False,
    model_family: str = "llama3",
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """

    sampler = Sampler(sampler_type=sampler_type, top_p=top_p, top_k=top_k)

    if model_family == "llama3":
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

        results, rngstates = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            sampler=sampler,
            genstates=genstates
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

        generated_tokens, _, rngstates = generate(  # type: ignore[operator]
            encoded_prompts = [tokens],
            model = model,
            max_tokens = max_seq_len,
            temperature = temperature,
            eos_id = tokenizer.eos_id,
            sampler = sampler,
            genstates = genstates,
            rng = rng
        )

        if genstates:
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
    output["sampler_type"] = sampler_type
    output["top_p"] = top_p
    output["top_k"] = top_k
    output["model_family"] = model_family
        
    # save the output of the factual generation
    with open(os.path.join(output_dir, "factual.json"), "w") as f:
        json.dump(output, f, indent=4)

    # prepare the intervention file
    output["ckpt_dir"] = ckpt_dir
    output["tokenizer_path"] = tokenizer_path
    output["start_from"] = 0
    output["token_list"] = {ind: tok for ind, tok in enumerate(results[0]["generation"]["token_list"])}
    with open(os.path.join(output_dir, "intervention_1.json"), "w") as f:
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

