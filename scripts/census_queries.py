import subprocess
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--model_family", default='llama3', choices=['llama3', 'mistral'])
args = parser.parse_args()
nproc_per_node = 1

model_family = args.model_family

if model_family=="llama3":
    seeds=[1094356, 2345656, 823845969964454]
    weights_path = "src/llama3/pretrained/Meta-Llama-3-8B-Instruct/"
    tokenizer_path = "src/llama3/pretrained/Meta-Llama-3-8B-Instruct/tokenizer.model"
elif model_family=="mistral":
    seeds=[34435326297321, 124638290, 823845969964454]
    weights_path = "src/mistral-inference/8B-Instruct/"
    tokenizer_path = "/src/mistral-inference/8B-Instruct/"



for i, seed in enumerate(seeds):
    params = {
        "ckpt_dir": weights_path,
        "tokenizer_path": tokenizer_path,
        "max_seq_len": 8192,
        "max_batch_size": 2,
        "seed": seed,
        "temperature": 0.8,
        "query": '"Generate census data of 50 fictional people."',
        "exp_name": f"{model_family}/census{i + 5}", # this is the name of the experiment -- results are saved under outputs/exp_name,
        "genstates": "True", # if True, the script will save the states of the random number generator,
        "system": '"Return only the following information: Age, Sex, Citizenship, Race, Ethnicity, Marital Status, Number of Children, Occupation, Income, Education. For Race, choose only between following options: White American, Black or African American, American Indian or Alaska Native, Asian American, Native Hawaiian or Other Pacific Islander, Other or Two or more races (multiracial). For Ethinicity, choose only between following options: Non-Hispanic/Latino or Hispanic/Latino. Return a list in json format delimited by \\"\`\`\`\\"."',
        "model_family": model_family
    }

    cmd = ['torchrun', '--nproc_per_node', f"{nproc_per_node}", 'src/single_query.py']
    for j, k_v in enumerate(params.items()):
        k, v = k_v
        if j >= 4:
            cmd.append(f"--{k}={v}")
        else:
            cmd.append(f"--{k}")
            cmd.append(str(v))
    cmd = ' '.join(cmd)
    subprocess.run(args=cmd, shell=True)
