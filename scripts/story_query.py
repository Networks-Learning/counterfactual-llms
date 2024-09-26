import subprocess
nproc_per_node = 1


params = {
    'query': "Tell me a fantasy story about a captain. The story should have either a happy or a sad ending.",
    'system': "Be creative and keep your response as short as possible.",
    'seed': 2,
    'temperature': 0.9,
    'max_seq_len': 2048,
    'max_batch_size': 2,
    'exp_name': "story-test",
    'sampler_type': "vocabulary", # "vocabulary" or "top-p position" or "top-p token" or "top-k token"
    'top_p': 0.9,   # this value doesn't matter if the respective sampler_type is not "top-p"
    'top_k': 5,    # this value doesn't matter if the respective sampler_type is not "top-k"
    'genstates': "True", 
    'ckpt_dir':"src/llama3/pretrained/Meta-Llama-3-8B-Instruct/",
    'tokenizer_path':"src/llama3/pretrained/Meta-Llama-3-8B-Instruct/tokenizer.model"
}

cmd = ['torchrun', '--nproc_per_node', f"{nproc_per_node}", 'src/single_query.py']
for k, v in params.items():
    cmd.append(f"--{k}")
    cmd.append(str(v))

subprocess.run(args=cmd)