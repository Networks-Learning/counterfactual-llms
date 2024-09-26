import subprocess
nproc_per_node = 1
seeds = [34435326297321, 124638290 ,823845969964454]
for i, seed in enumerate(seeds):
    params = {
        "ckpt_dir":"src/llama3/pretrained/Meta-Llama-3-8B-Instruct/",
        "tokenizer_path":"src/llama3/pretrained/Meta-Llama-3-8B-Instruct/tokenizer.model",
        "max_seq_len": 8192,
        "max_batch_size": 2,
        "seed": seed,
        "temperature": 0.8,
        "query": '"Generate census data of 50 fictional people."',
        "exp_name": f"census{i + 1}", # this is the name of the experiment -- results are saved under outputs/exp_name,
        "genstates": "True", # if True, the script will save the states of the random number generator,
        "system": '"Return only the following information: Name, Age, Sex, Citizenship, Race, Ethnicity, Marital Status, Number of Children, Occupation, Income, Education. For Race, choose only between following options: White American, Black or African American, American Indian or Alaska Native, Asian American, Native Hawaiian or Other Pacific Islander, Other or Two or more races (multiracial). For Ethinicity, choose only between following options: Non-Hispanic/Latino or Hispanic/Latino. Return a list in json format delimited by \\"\`\`\`\\"."',
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
