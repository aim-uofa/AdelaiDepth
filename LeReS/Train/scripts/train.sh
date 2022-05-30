
export PYTHONPATH=../../Train:${PYTHONPATH}
# export CUDA_VISIBLE_DEVICES=0,1,2,3

TIME=`date +%Y-%m-%d_%H-%M-%S`

LOG="./$TIME.txt"

python ../tools/train.py \
--dataroot datasets \
--dataset_list taskonomy DIML_GANet DiverseDepth \
--batchsize 16 \
--base_lr 0.001 \
--use_tfboard \
--thread 4 \
--loss_mode _ssil_vnl_ranking_ \
--epoch 50 \
--lr_scheduler_multiepochs 10 25 40 \
$1 2>&1 | tee $LOG
