#!/bin/bash -l

#$ -P ace-ig          # SCC project name
#$ -l h_rt=13:00:00   # Specify the hard time limit for the job
#$ -N with_pair_loss   # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1
#$ -l gpu_c=6.0

module load python3/3.8.10
export PYTHONPATH=/projectnb/ace-ig/jw_python/lib/python3.8.10/site-packages:$PYTHONPATH
module load pytorch/1.13.1


python3 ./code/project/main.py \
	--tensor_board_logger="./tensorboard/" \
	--dataset="ACE" \
	--experiment_name="ACE" \
	--classifier_latent_dim=24 \
	--batch_size=128 \
	--learning_rate=5e-5 \
	--test_fold=0 \
	--run_time=0 \
	--n_epochs=100 \
	--hidden_dim_qk=32 \
	--hidden_dim_q=32 \
	--hidden_dim_k=4 \
	--hidden_dim_v=8 \
	--relu_at_coattention \
	--normalization="batch" \
	--bernoulli_probability=1e-2 \
	--sparsity_loss_weight=1e-6 \
	--soft_sign_constant=0.5 \
	--contrastive_metric="L1" \
	--diff_pair_loss \
	--pair_loss_weight=0.0001 \
	--not_write_tensorboard

