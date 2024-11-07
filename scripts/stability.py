import argparse
import subprocess

# Constant parameters
# llama3
ckpt_path=f"src/llama3/pretrained/Meta-Llama-3-8B-Instruct/"
tokenizer_path=f"{ckpt_path}/tokenizer.model"
model_family = 'llama3'

# mistral 
# ckpt_path = f"src/mistral-inference/8B-Instruct/"
# tokenizer_path = f"{ckpt_path}/src/mistral-inference/8B-Instruct/"
# model_family = 'mistral'

seed = 42
system = "Keep your replies short and to the point."
top_p = [0.75,0.9,0.95,0.99,0.999]
top_k = [2,3,5,10,100]
temperature = [0.0,0.2,0.4,0.6,0.8,1.0]
input_file = f"data/questions.parquet"
output_dir = "outputs/stability/mistral" # this is the name of the experiment -- results are saved under output_dir
chunk_size = 25
num_interventions = 2
categorical = False

parser=argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default=ckpt_path)
parser.add_argument("--tokenizer_path", default=tokenizer_path)  
parser.add_argument("--max_seq_len", default=2048, type=int)
parser.add_argument("--max_batch_size", default=2, type=int)
parser.add_argument("--seed", default=seed, type=int, 
                    help="the seed for the initialization of the random number generator") 
parser.add_argument("--input_file", default=input_file,
                    help="path to file containing input queries")
parser.add_argument("--output_dir", default=output_dir,
                    help="path to output directory")
parser.add_argument("--output_file_params", default="params",
                    help="file name to store experiment parameters")
parser.add_argument("--output_file_responses", default="stability_strings",
                    help="file name to store all model responses")
parser.add_argument("--system", default=system, 
                    help="the system prompt you want to use")
parser.add_argument("--temperature", default=temperature,
                    help="the temperature parameter of the model")
parser.add_argument("--top_p",default=top_p,
                    help="the value of p for top-p samplers")
parser.add_argument("--top_k",default=top_k,
                    help="the value of k for the top-k sampler")
parser.add_argument("--chunk_size",default=chunk_size,
                    help="number of queries to be written to file at once")
parser.add_argument("--num_interventions",default=num_interventions,
                    help="number of interventions per query")
parser.add_argument("--top_p_intervention", default=0.9,
                    help="top-p value to select the post-intervention token")
parser.add_argument("--intervention_seed", default=seed,
                    help="Seed to select the post-intervention token")
parser.add_argument("--intervention_position_seed", default=seed,
                    help="Seed to select the intervention position")
parser.add_argument("--model_family", default=model_family, choices=['llama3', 'mistral'],
                    help="Please select the model family")
parser.add_argument("--categorical", default=False,
                    help="If true, runs inverse transform sampler")

args = parser.parse_args()
if args.model_family == 'llama3':
    args.ckpt_dir = f"src/llama3/pretrained/Meta-Llama-3-8B-Instruct/"
    args.tokenizer_path = f"{args.ckpt_dir}/tokenizer.model"
    args.output_dir = "outputs/stability/llama3"
elif args.model_family == 'mistral':
    args.ckpt_dir = f"src/mistral-inference/8B-Instruct/"
    args.tokenizer_path = f"{args.ckpt_dir}/tokenizer.model"
    args.output_dir = "outputs/stability/mistral"
if args.categorical == 'True':
    args.top_p=[]
    args.top_k=[]
    args.output_dir="outputs/stability/categorical"


nproc_per_node=1
cmd = ['torchrun', '--nproc_per_node', f"{nproc_per_node}", 'src/stability.py']
for k, v in vars(args).items():
    cmd.append(f"--{k}")
    cmd.append(str(v))

print(cmd)
subprocess.run(args=cmd)


