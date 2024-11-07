import subprocess
nproc_per_node = 1

params = {
    'max_seq_len': 2048,
    'max_batch_size': 2,
    'exp_name': "story-test",
    'iteration': 1,
    'prior': False,
}

cmd = ['torchrun', '--nproc_per_node', f"{nproc_per_node}", 'src/cf_query.py']
for k, v in params.items():
    cmd.append(f"--{k}")
    cmd.append(str(v))

subprocess.run(args=cmd)