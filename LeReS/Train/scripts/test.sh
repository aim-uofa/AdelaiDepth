
export PYTHONPATH=../../Train:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

TIME=`date +%Y-%m-%d_%H-%M-%S`

LOG="./$TIME.txt"

python ../tools/test_multiauxiv2_nyu.py \
--dataroot ./datasets \
--batchsize 1 \
--load_ckpt path_to_ckpt.pth \
$1 2>&1 | tee $LOG
