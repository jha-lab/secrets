#!/bin/bash


debug_exp=0

module purge
module load anaconda3/2021.11


#### setting computing environment parameters ####

use_adroit=0
if [[ $HOSTNAME == *"adroit"* ]]; then
	use_adroit=1
fi
use_tiger=0
if [[ $HOSTNAME == *"tiger"* ]]; then
	use_tiger=1
fi
use_della=0
if [[ $HOSTNAME == *"della"* ]]; then
	use_della=1
fi

#### setting path variables ####
prefix="/scratch/gpfs/slala"
if [[ $use_adroit == 1 ]]; then
	prefix=/scratch/network/slala
fi 

conda_env=rct_env
if [[ $use_adroit == 1 ]]; then
	conda_env=/scratch/network/slala/rct_env
fi 
conda activate $conda_env


#### experimental/algorithmic parameters ####

# number of trials
seed_start=0
num_seeds=1000
if [[ $debug_exp == 1 ]]; then
	num_seeds=10
fi 
seed_end=$(($num_seeds-1))


outcome_type=twa 
use_unbiased_var=0 # historical reasons
num_tune_seeds=1000 # bootstraps

if [[ $debug_exp == 1 ]]; then
	num_tune_seeds=1000
fi

arm_size=30

use_mp=1
num_checkpoints=50
print_every=1000
if [[ $debug_exp == 1 ]]; then
	num_checkpoints=2
	print_every=1
fi 

#### computing env details ####

if [[ $use_adroit == 1 ]]; then
	num_workers=32 #18
	mem_per_cpu=4
	gpus=0
	time='16:00:00' # includes queue time
	module_name=anaconda3/2021.11
elif [[ $use_tiger == 1 ]]; then
	num_workers=7 #18
	mem_per_cpu=4
	gpus=0
	time='00:30:00' # includes queue time
	module_name=anaconda3/2021.11
elif [[ $use_della == 1 ]]; then
	num_workers=28
	mem_per_cpu=4
	gpus=0
	time='00:30:00'
	module_name=anaconda3/2021.11
fi 

if [[ $debug_exp == 1 ]]; then
	num_workers=7
	time='00:30:00'
fi 


# prepare results directory
results_path_prefix="/scratch/gpfs/slala/secrets/mgtx"
if [[ $HOSTNAME == *"adroit"* ]]; then
	results_path_prefix="/scratch/network/slala/secrets/mgtx"
fi 
folder_names=(arm_size)

for folder_name in "${folder_names[@]}"
do
	results_path_prefix="$results_path_prefix/${folder_name}_${!folder_name}"
done 

mkdir -p $results_path_prefix


# estimate_ITEs across the trials, for both control/treatment arms (indexed by eval_alpha)
# for eval_alpha in $(seq 0 1)
# do
# 	results_path=$results_path_prefix/eval_alpha_${eval_alpha}

# 	mkdir -p $results_path

# 	python gen_job_slurm_mgtx.py \
# 	--results_path $results_path \
# 	--num_workers $num_workers \
# 	--mem_per_cpu $mem_per_cpu \
# 	--gpus $gpus \
# 	--time $time \
# 	--module_name $module_name \
# 	--conda_env $conda_env \
# 	--eval_alpha $eval_alpha \
# 	--outcome_type $outcome_type \
# 	--use_unbiased_var $use_unbiased_var \
# 	--seed_start $seed_start \
# 	--seed_end $seed_end \
# 	--arm_size $arm_size \
# 	--use_mp $use_mp \
# 	--num_checkpoints $num_checkpoints \
# 	--print_every $print_every \
# 	2> $results_path/gen_job_slurm.err 1> $results_path/gen_job_slurm.out

# 	slurm_job_id=$(sbatch $results_path/job.slurm)
# 	# echo $slurm_job_id
# 	slurm_job_file="${results_path}/slurm_job_id.txt"
# 	# echo "$slurm_job_id" > "$slurm_job_file" 
# 	# echo $slurm_job_file 
# done 



# run_hypothesis_test: sample_null

# divvy jobs over cores to increase parallelism
num_data_seeds_per_job=100
if [[ $debug_exp == 1 ]]; then
	num_data_seeds_per_job=10
fi 
num_data_seed_jobs=$(($num_seeds/$num_data_seeds_per_job))
num_data_seed_jobs=$(($num_data_seed_jobs-1))
idsjs=$(seq 0 $num_data_seed_jobs)

results_path=$results_path_prefix/num_tune_seeds_${num_tune_seeds}
mkdir -p $results_path

job_ids=()
for idsj in ${idsjs[@]}
do
	dss=$(($idsj*$num_data_seeds_per_job))
	dse=$((($idsj+1)*$num_data_seeds_per_job))
	dse=$(($dse-1))

	python gen_job_slurm_mgtx.py \
	--results_path $results_path \
	--num_workers $num_workers \
	--mem_per_cpu $mem_per_cpu \
	--gpus $gpus \
	--time $time \
	--module_name $module_name \
	--conda_env $conda_env \
	--eval_alpha 1 \
	--outcome_type $outcome_type \
	--use_unbiased_var $use_unbiased_var \
	--tune_cv_per_data_seed 1 \
	--num_tune_seeds $num_tune_seeds \
	--data_seed_start $dss \
	--data_seed_end $dse \
	--arm_size $arm_size \
	--use_mp $use_mp \
	--num_checkpoints $num_checkpoints \
	--print_every $print_every \
	2> $results_path/gen_job_slurm.err 1> $results_path/gen_job_slurm.out

	slurm_job_id=$(sbatch $results_path/job.slurm)
	# echo $slurm_job_id
	slurm_job_file="${results_path}/slurm_job_id.txt"
	# echo "$slurm_job_id" > "$slurm_job_file" 
	# echo $slurm_job_file 

	job_ids+=(${slurm_job_id##* })
done 

# tune_critical_value, run_test
cv_results_path=$results_path_prefix/cv
mkdir -p $cv_results_path

python gen_job_slurm_critical_value.py \
	--num_workers 2 \
	--mem_per_cpu 4 \
	--gpus 0 \
	--time $time \
	--module_name $module_name \
	--conda_env $conda_env \
	--prefix_file $results_path_prefix \
	--data_seed_prefix_dir $results_path \
	--results_path $cv_results_path \
	--arm_size $arm_size \
	--data_seed_start $seed_start \
	--data_seed_end $seed_end \
2> $cv_results_path/gen_job_slurm.err 1> $cv_results_path/gen_job_slurm.out

slurm_job_id=$(sbatch -d afterok:${job_ids[0]}:${job_ids[1]} $cv_results_path/job.slurm)
# slurm_job_id=$(sbatch -d afterok:$slurm_job_id $cv_results_path/job.slurm)
# echo $slurm_job_id
slurm_job_file="${cv_results_path}/slurm_job_id.txt"
# echo "$slurm_job_id" > "$slurm_job_file" 
# echo $slurm_job_file
 






