import subprocess
import argparse

nproc_per_node = 1
parser = argparse.ArgumentParser()
parser.add_argument('exp_name', help='experiment name')
parser.add_argument('iteration', help='iteration number', type=int)
args = parser.parse_args()

params = {
    'query': "Tell me a fantasy story about a captain. The story should have either a happy or a sad ending.",
    'system': "Be creative and keep your response as short as possible.",
    'max_seq_len': 2048,
    'max_batch_size': 2,
    'exp_name': args.exp_name,
    'iteration': args.iteration,
    'ckpt_dir':"src/llama3/pretrained/Meta-Llama-3-8B-Instruct/",
    'tokenizer_path':"src/llama3/pretrained/Meta-Llama-3-8B-Instruct/tokenizer.model"
}

cmd = ['torchrun', '--nproc_per_node', f"{nproc_per_node}", 'src/cf_query.py']
for k, v in params.items():
    cmd.append(f"--{k}")
    cmd.append(str(v))

subprocess.run(args=cmd)