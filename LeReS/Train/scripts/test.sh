
export PYTHONPATH=../../Train:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

TIME=`date +%Y-%m-%d_%H-%M-%S`

LOG="./$TIME.txt"

python ../tools/test_multiauxiv2_nyu.py \
--dataroot ./datasets \
--batchsize 1 \
--load_ckpt /home/gk-ai/桌面/AdelaiDepth/LeReS/Train/scripts/output11111/May29-21-21-39_gk-ai-ustc/ckpt/epoch0_step10.pth \
$1 2>&1 | tee $LOG
