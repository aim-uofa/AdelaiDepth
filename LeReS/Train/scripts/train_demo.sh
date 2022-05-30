
# export PYTHONPATH=../../Train:${PYTHONPATH}
export PYTHONPATH=../
# export CUDA_VISIBLE_DEVICES=0,1,2,3

python ../tools/train.py \
--dataroot datasets \
--backbone resnet50 \
--dataset_list demo \
--batchsize 2 \
--base_lr 0.1 \
--use_tfboard \
--thread 4 \
--loss_mode _ranking-edge_pairwise-normal-regress-edge_msgil-normal_meanstd-tanh_pairwise-normal-regress-plane_ranking-edge-auxi_meanstd-tanh-auxi_ \
--epoch 100 \
--lr_scheduler_multiepochs 10 25 40 \
--output_dir output11111 \
--val_step 10 \
--snapshot_iters 10 \
--log_interval 5 

#  DiverseDepth