import subprocess
nproc_per_node=1
cmd = ['torchrun', '--nproc_per_node', f"{nproc_per_node}", 'src/bias.py']
subprocess.run(args=cmd)