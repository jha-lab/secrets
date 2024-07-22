import argparse
import os 
import sys
sys.path.append("..")
import src
from src import syn_ctrl_mgtx

def parse_str_helper(inp_str):
	if inp_str == "None":
		return None
	return inp_str

parser=argparse.ArgumentParser()
parser.add_argument("--use_della", type=int,default=0)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--mem_per_cpu", type=int, default=4)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--time", type=str)
parser.add_argument("--module_name", type=str, default="anaconda3/2021.11")
parser.add_argument("--conda_env", type=str, default="reder_env")

syn_ctrl_mgtx.add_arguments(parser)

args = parser.parse_args()

out_path=os.path.join(args.results_path, 'slurm.out')
err_path=os.path.join(args.results_path,'slurm.err')

with open(os.path.join(args.results_path,"job.slurm"), 'w') as f:
	f.write("#!/bin/bash\n")
	f.write("#SBATCH --job-name=syn_ctrl # create a short name for your job\n")
	f.write("#SBATCH -o %s\n" %out_path)
	f.write("#SBATCH -e %s\n" %err_path)
	f.write("#SBATCH --nodes=1                # node count\n")
	f.write("#SBATCH --ntasks=1               # total number of tasks across all nodes\n")
	f.write("#SBATCH --cpus-per-task=%d        # cpu-cores per task (>1 if multi-threaded tasks)\n" %args.num_workers)
	f.write("#SBATCH --mem-per-cpu=%dG         # memory per cpu-core (4G is default)\n"%args.mem_per_cpu)
	if args.gpus>0: f.write("#SBATCH --gres=gpu:%d             # number of gpus per node\n" % args.gpus)
	f.write("#SBATCH --time=" + args.time + "          # total run time limit (e.g., HH:MM:SS or days-hours)\n")
	# f.write("#SBATCH mail-type=begin        # send mail when process begins\n")
	# f.write("#SBATCH mail-type=end          # send email when job ends\n")
	# f.write("#SBATCH mail-user=slala@princeton.edu\n")
	if args.use_della: f.write("#SBATCH --exclude=della-r4c[1-4]n[1-16],della-r1c[3-4]n[1-16],della-r3c[1-4]n[1-16]") # index skylake and above
	f.write("\n")
	f.write("module purge\n")
	f.write("module load %s\n"%args.module_name)
	f.write("conda activate %s\n" %args.conda_env)
	f.write("\n")
	f.write("srun -u python ../src/syn_ctrl_mgtx.py \\\n")	

	for key, value in vars(args).items():
		# ignore the SLURM
		if key in ["num_workers", "mem_per_cpu", "gpus", "time", "module_name", "conda_env", "cpu_type", "use_della"]:
			continue

		f.write("--%s %s \\\n"%(key,str(value)))

