import subprocess
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--i", default=1)
parser.add_argument("--direct", default=False)
parser.add_argument("--prior", default=False)
parser.add_argument("--model_family", default='llama3', choices=['llama3', 'mistral'])
parser.add_argument("--attribute_to_intervene", default='Sex', choices=['Sex', 'Race'])
args = parser.parse_args()

nproc_per_node=1
cmd = ['torchrun', '--nproc_per_node', f"{nproc_per_node}", 'src/bias.py']
for k, v in vars(args).items():
    cmd.append(f"--{k}")
    cmd.append(str(v))
subprocess.run(args=cmd)