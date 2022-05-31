export PYTHONPATH=../../Train:${PYTHONPATH}

python ../tools/train.py \
--dataroot datasets \
--backbone resnet50 \
--dataset_list demo \
--batchsize 2 \
--base_lr 0.001 \
--use_tfboard \
--thread 4 \
--loss_mode _ranking-edge_pairwise-normal-regress-edge_msgil-normal_meanstd-tanh_pairwise-normal-regress-plane_ranking-edge-auxi_meanstd-tanh-auxi_ \
--epoch 2 \
--lr_scheduler_multiepochs 1 \
--val_step 50 \
--snapshot_iters 50 \
--log_interval 5 
