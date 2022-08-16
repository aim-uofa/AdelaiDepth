import math
import traceback
import errno
import os
import os.path as osp
import json
import torch.distributed as dist
import torch.multiprocessing as mp

from multiprocessing.sharedctypes import Value
from data.load_dataset_distributed import MultipleDataLoaderDistributed
from lib.models.multi_depth_model_auxiv2 import *
from lib.configs.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_rel_depth_err, recover_metric_depth
from lib.utils.lr_scheduler_custom import make_lr_scheduler
from lib.utils.comm import is_pytorch_1_1_0_or_later, get_world_size
from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_distributed_logger, SmoothedValue
from tools.parse_arg_base import print_options
from tools.parse_arg_train import TrainOptions
from tools.parse_arg_val import ValOptions


def main_process(dist, rank) -> bool:
    return not dist or (dist and rank == 0)

def increase_sample_ratio_steps(step, base_ratio=0.1, step_size=10000):
    ratio = min(base_ratio * (int(step / step_size) + 1), 1.0)
    return ratio

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def val(val_dataloader, model):
    """
    Validate the model.
    """
    print('validating...')
    smoothed_absRel = SmoothedValue(len(val_dataloader))
    smoothed_whdr = SmoothedValue(len(val_dataloader))
    smoothed_criteria = {'err_absRel': smoothed_absRel, 'err_whdr': smoothed_whdr}
    for i, data in enumerate(val_dataloader):
        out = model.module.inference(data)
        pred_depth = torch.squeeze(out['pred_depth'])

        pred_depth_resize = cv2.resize(pred_depth.cpu().numpy(), (torch.squeeze(data['gt_depth']).shape[1], torch.squeeze(data['gt_depth']).shape[0]))
        pred_depth_metric = recover_metric_depth(pred_depth_resize, data['gt_depth'])
        smoothed_criteria = validate_rel_depth_err(pred_depth_metric, data['gt_depth'], smoothed_criteria, scale=1.0)
    return {'abs_rel': smoothed_criteria['err_absRel'].GetGlobalAverageValue(),
            'whdr': smoothed_criteria['err_whdr'].GetGlobalAverageValue()}


def do_train(train_dataloader, val_dataloader, train_args,
             model, save_to_disk,
             scheduler, optimizer, val_err,
             logger, tblogger=None):
    # training status for logging
    if save_to_disk:
        training_stats = TrainingStats(train_args, cfg.TRAIN.LOG_INTERVAL, tblogger if train_args.use_tfboard else None)

    dataloader_iterator = iter(train_dataloader)
    start_step = train_args.start_step
    total_iters = cfg.TRAIN.MAX_ITER
    train_datasize = train_dataloader.batch_sampler.sampler.total_sampled_size

    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    tmp_i = 0
    try:
        for step in range(start_step, total_iters):

            if step % train_args.sample_ratio_steps == 0 and step != 0:
                sample_ratio = increase_sample_ratio_steps(step, base_ratio=train_args.sample_start_ratio, step_size=train_args.sample_ratio_steps)
                train_dataloader, curr_sample_size = MultipleDataLoaderDistributed(train_args, sample_ratio=sample_ratio)
                dataloader_iterator = iter(train_dataloader)
                logger.info('Sample ratio: %02f, current sampled datasize: %d' % (sample_ratio, np.sum(curr_sample_size)))

            epoch = int(step * train_args.batchsize*train_args.world_size / train_datasize)
            if save_to_disk:
                training_stats.IterTic()

            # get the next data batch
            try:
                data = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_dataloader)
                data = next(dataloader_iterator)

            out = model(data)
            losses_dict = out['losses']
            optimizer.optim(losses_dict)

            #################Check data loading######################
            # tmp_path_base = '/home/yvan/DeepLearning/Depth/DiverseDepth-github/DiverseDepth/datasets/x/'
            # rgb = data['A'][1, ...].permute(1, 2, 0).squeeze()
            # rgb =rgb * torch.tensor(cfg.DATASET.RGB_PIXEL_VARS)[None, None, :] + torch.tensor(cfg.DATASET.RGB_PIXEL_MEANS)[None, None, :]
            # rgb = rgb * 255
            # rgb = rgb.cpu().numpy().astype(np.uint8)
            # depth = (data['B'][1, ...].squeeze().cpu().numpy()*1000)
            # depth[depth<0] = 0
            # depth = depth.astype(np.uint16)
            # plt.imsave(tmp_path_base+'%04d_r.jpg' % tmp_i, rgb)
            # plt.imsave(tmp_path_base+'%04d_d.png' % tmp_i, depth, cmap='rainbow')
            # tmp_i +=1
            #########################################################

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(losses_dict)

            scheduler.step()
            if save_to_disk:
                training_stats.UpdateIterStats(loss_dict_reduced)
                training_stats.IterToc()
                training_stats.LogIterStats(step, epoch, optimizer.optimizer, val_err[0])

            # validate the model
            if step % cfg.TRAIN.VAL_STEP == 0 and val_dataloader is not None and step != 0:
                model.eval()
                val_err[0] = val(val_dataloader, model)
                # training mode
                model.train()
            # save checkpoint
            if step % cfg.TRAIN.SNAPSHOT_ITERS == 0 and step != 0 and save_to_disk:
                save_ckpt(train_args, step, epoch, model, optimizer.optimizer, scheduler, val_err[0])

    except (RuntimeError, KeyboardInterrupt):
        stack_trace = traceback.format_exc()
        print(stack_trace)
    finally:
        if train_args.use_tfboard and main_process(dist=train_args.distributed, rank=train_args.global_rank):
            tblogger.close()


def main_worker(local_rank: int, ngpus_per_node: int, train_args, val_args):
    train_args.global_rank = train_args.node_rank * ngpus_per_node + local_rank
    train_args.local_rank = local_rank
    val_args.global_rank = train_args.global_rank
    val_args.local_rank = local_rank
    merge_cfg_from_file(train_args)

    global logger
    # Set logger
    log_output_dir = cfg.TRAIN.LOG_DIR
    if log_output_dir:
        try:
            os.makedirs(log_output_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    logger = setup_distributed_logger("lib", log_output_dir, local_rank, cfg.TRAIN.RUN_NAME + '.txt')
    tblogger = None
    if train_args.use_tfboard and  local_rank == 0:
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)

    # init
    if train_args.distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',
                                init_method=train_args.dist_url,
                                world_size=train_args.world_size,
                                rank=train_args.global_rank)


    # load model
    model = RelDepthModel()
    if train_args.world_size>1:
        assert is_pytorch_1_1_0_or_later(), \
            "SyncBatchNorm is only available in pytorch >= 1.1.0"
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info('Using SyncBN!')


    if train_args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[local_rank], output_device=local_rank)
    else:
        model = torch.nn.DataParallel(model.cuda())

    val_err = [{'abs_rel': 0, 'whdr': 0}]

    # Print configs and logs
    print_options(train_args, logger)

    # training and validation dataloader
    train_dataloader, train_sample_size = MultipleDataLoaderDistributed(train_args)
    val_dataloader, val_sample_size = MultipleDataLoaderDistributed(val_args)
    cfg.TRAIN.LR_SCHEDULER_MULTISTEPS = np.array(train_args.lr_scheduler_multiepochs) * math.ceil(np.sum(train_sample_size)/ (train_args.world_size * train_args.batchsize))

    # Optimizer
    optimizer = ModelOptimizer(model)
    # lr_optim_lambda = lambda iter: (1.0 - iter / (float(total_iters))) ** 0.9
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.optimizer, lr_lambda=lr_optim_lambda)
    # lr_scheduler_step = 15000
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optimizer, step_size=lr_scheduler_step, gamma=0.9)
    scheduler = make_lr_scheduler(cfg=cfg, optimizer=optimizer.optimizer)

    if train_args.load_ckpt:
        load_ckpt(train_args, model, optimizer.optimizer, scheduler, val_err)
        # obtain the current sample ratio
        sample_ratio = increase_sample_ratio_steps(train_args.start_step, base_ratio=train_args.sample_start_ratio,
                                                   step_size=train_args.sample_ratio_steps)
        # reconstruct the train_dataloader with the new sample_ratio
        train_dataloader, train_sample_size = MultipleDataLoaderDistributed(train_args, sample_ratio=sample_ratio)


    total_iters = math.ceil(np.sum(train_sample_size)/ (train_args.world_size * train_args.batchsize)) * train_args.epoch
    cfg.TRAIN.MAX_ITER = total_iters
    cfg.TRAIN.GPU_NUM = train_args.world_size
    print_configs(cfg)

    save_to_disk = main_process(train_args.distributed, local_rank)

    do_train(train_dataloader,
             val_dataloader,
             train_args,
             model,
             save_to_disk,
             scheduler,
             optimizer,
             val_err,
             logger,
             tblogger)

def main():
    # Train args
    train_opt = TrainOptions()
    train_args = train_opt.parse()

    # Validation args
    val_opt = ValOptions()
    val_args = val_opt.parse()
    val_args.batchsize = 1
    val_args.thread = 0

    # Validation datasets
    val_datasets = []
    for dataset_name in val_args.dataset_list:
        val_annos_path = osp.join(cfg.ROOT_DIR, val_args.dataroot, dataset_name, 'annotations', 'val_annotations.json')
        if not osp.exists(val_annos_path):
            continue
        with open(val_annos_path, 'r') as f:
            anno = json.load(f)[0]
            if 'depth_path' not in anno:
                continue
            depth_path_demo = osp.join(cfg.ROOT_DIR, val_args.dataroot, anno['depth_path'])
            depth_demo = cv2.imread(depth_path_demo, -1)
            if depth_demo is not None:
                val_datasets.append(dataset_name)
    val_args.dataset_list = val_datasets

    print('Using PyTorch version: ', torch.__version__, torch.version.cuda)
    ngpus_per_node = torch.cuda.device_count()
    train_args.world_size = ngpus_per_node * train_args.nnodes
    val_args.world_size = ngpus_per_node * train_args.nnodes
    train_args.distributed = ngpus_per_node > 1

    # Randomize args.dist_url to avoid conflicts on same machine
    train_args.dist_url = train_args.dist_url + str(os.getpid() % 100).zfill(2)

    if train_args.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, train_args, val_args))
    else:
        main_worker(0, ngpus_per_node, train_args, val_args)


if __name__=='__main__':
    main()
