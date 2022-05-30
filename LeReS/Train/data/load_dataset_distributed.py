import importlib
import math
import logging
import numpy as np
import torch.utils.data

import torch.distributed as dist

logger = logging.getLogger(__name__)

# class MultipleDataLoaderDistributed():
#     def __init__(self, opt, sample_ratio=0.1):
#         self.opt = opt
#         self.multi_datasets, self.dataset_indices_list = create_multiple_dataset(opt)
#         self.datasizes = [len(dataset) for dataset in self.multi_datasets]
#         self.merged_dataset = torch.utils.data.ConcatDataset(self.multi_datasets)
#         #self.custom_multi_sampler_dist = CustomerMultiDataSamples(self.dataset_indices_list, sample_ratio, self.opt)
#         self.custom_multi_sampler_dist = CustomerMultiDataSampler(opt, self.merged_dataset, opt.world_size, opt.phase)
#         self.curr_sample_size = self.custom_multi_sampler_dist.num_samples
#         self.dist_sample_size = self.custom_multi_sampler_dist.num_dist_samples
#         self.dataloader = torch.utils.data.DataLoader(
#             dataset=self.merged_dataset,
#             batch_size=opt.batchsize,
#             num_workers=opt.thread,
#             sampler=self.custom_multi_sampler_dist,
#             drop_last=True)
#
#     def load_data(self):
#         return self
#
#     def __len__(self):
#         return np.sum(self.datasizes)
#
#     def __iter__(self):
#         for i, data in enumerate(self.dataloader):
#             if i * self.opt.batchsize >= float("inf"):
#                 break
#             yield data

def MultipleDataLoaderDistributed(opt, sample_ratio=1):
        opt = opt
        #multi_datasets, dataset_indices_list = create_multiple_dataset(opt)
        multi_datasets = create_multiple_dataset(opt)
        #multi_datasizes = [len(dataset) for dataset in multi_datasets]
        merged_dataset = torch.utils.data.ConcatDataset(multi_datasets)
        #custom_multi_sampler_dist = CustomerMultiDataSamples(dataset_indices_list, sample_ratio, opt)
        custom_multi_sampler_dist = CustomerMultiDataSampler(opt, merged_dataset, opt.world_size, opt.phase)
        curr_sample_size = custom_multi_sampler_dist.num_samples
        dist_sample_size = custom_multi_sampler_dist.num_dist_samples
        dataloader = torch.utils.data.DataLoader(
            dataset=merged_dataset,
            batch_size=opt.batchsize,
            num_workers=opt.thread,
            sampler=custom_multi_sampler_dist,
            drop_last=True)

        return dataloader, curr_sample_size

class CustomerMultiDataSampler(torch.utils.data.Sampler):
    """
    Construct a sample method. Sample former ratio_samples of datasets randomly.
    """

    def __init__(self, args, multi_dataset, num_replicas, split, sample_ratio=1.0):
        self.args = args
        self.num_replicas = num_replicas
        self.phase = split
        self.rank = args.global_rank
        #self.logger = logger

        self.multi_dataset = multi_dataset
        self.create_samplers()

        self.num_indices = np.array([len(i) for i in self.extended_indices_list])
        #self.num_samples = self.num_indices.astype(np.uint32)
        self.num_samples = (self.num_indices * sample_ratio).astype(np.uint32)
        self.max_indices = np.array([max(i) for i in self.extended_indices_list])
        self.total_sampled_size = np.sum(self.num_samples)
        self.num_dist_samples = int(
            math.ceil(self.total_sampled_size * 1.0 / self.num_replicas)
        )
        self.total_dist_size = self.num_dist_samples * self.num_replicas

        logstr = ",".join(["%s sampled data size: %d" % (args.dataset_list[i], self.num_samples[i])
                              for i in range(self.num_indices.size)]
                         )
        logger.info(logstr)

    def __iter__(self):
        self.create_samplers()
        cum_sum = np.cumsum(np.append([0], self.max_indices))
        indices_array = [[self.extended_indices_list[data_i][i] + cum_sum[data_i]
                              for i in range(int(num))]for data_i, num in enumerate(self.num_samples)]

        if "train" in self.phase:
            # data list is mapped to the order [A, B, C, A, B, C....]
            indices_array = np.array(indices_array).transpose(1, 0).reshape(-1)
        else:
            indices_array = np.concatenate(indices_array[:])

        # add extra samples to make it evenly divisible
        diff_size = int(self.total_dist_size - self.total_sampled_size)
        if diff_size > 0:
            extended_indices_dist = np.append(indices_array, indices_array[:diff_size])
        else:
            extended_indices_dist = indices_array
        assert extended_indices_dist.size == self.total_dist_size

        # subsample
        offset = self.num_dist_samples * self.rank
        rank_indices = extended_indices_dist[offset : offset + self.num_dist_samples]
        assert rank_indices.size == self.num_dist_samples
        for id in rank_indices:
            yield id

    def __len__(self):
        return self.total_sampled_size

    def create_samplers(self):
        self.extended_indices_list = []

        dataset_indices_lists = []
        indices_len = []
        datasets_num = len(self.multi_dataset.datasets)
        for dataset_i in self.multi_dataset.datasets:
            # The list of indices of each dataset
            dataset_indices_lists.append(np.random.permutation(np.arange(len(dataset_i.curriculum_list))))
            indices_len.append(len(dataset_i.curriculum_list))

        # the max size of all datasets
        max_len = np.max(indices_len)

        if "train" == self.phase:
            for data_list in dataset_indices_lists:
                cp = max_len // data_list.size
                size_i = data_list.size
                tmp = data_list
                for i in range(cp-1):
                    tmp = np.concatenate((tmp, np.random.permutation(data_list)), axis=None)
                tmp = np.concatenate((tmp, np.random.choice(data_list, max_len % size_i, replace=False)), axis=None)
                self.extended_indices_list.append(list(tmp))
        else:
            self.extended_indices_list = dataset_indices_lists
        logstr = "\n".join(["Split %s, %s: %d -(extend to)-> %d" %
                                (self.phase, self.args.dataset_list[i], len(dataset_indices_lists[i]),
                                 len(self.extended_indices_list[i]))
                                for i in range(datasets_num)]
                               )
        logger.info(logstr)



class CustomerMultiDataSamples(torch.utils.data.Sampler):
    """
    Construct a sample method. Sample former ratio_samples of datasets randomly.
    """
    def __init__(self, multi_data_indices, ratio_samples, opt, rank=None, num_replicas=None):
        logger = logging.getLogger(__name__)

        self.multi_data_indices = multi_data_indices
        self.num_indices = np.array([len(i) for i in self.multi_data_indices])
        self.num_samples = (self.num_indices * ratio_samples).astype(np.uint32)
        self.max_indices = np.array([max(i) for i in self.multi_data_indices])
        self.total_sampled_size = np.sum(self.num_samples)
        self.phase = opt.phase
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        self.num_replicas = num_replicas
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.rank = rank
        self.num_dist_samples = int(math.ceil(self.total_sampled_size * 1.0 / self.num_replicas))
        self.total_dist_size = self.num_dist_samples * self.num_replicas
        logger.info('Sample %02f, sampled dataset sizes are %s' % (ratio_samples, ','.join(map(str, self.num_samples))))

    def __iter__(self):
        cum_sum = np.cumsum(np.append([0], self.max_indices))
        if 'train' in self.phase:
            indices_array = [[self.multi_data_indices[idx][i] + cum_sum[idx] for i in torch.randperm(int(num))] for
                             idx, num in
                             enumerate(self.num_samples)]
        else:
            indices_array = [[self.multi_data_indices[idx][i] + cum_sum[idx] for i in range(int(num))] for
                             idx, num in enumerate(self.num_samples)]
        if 'train' in self.phase:
            # data list is reshaped in [A, B, C, A, B, C....]
            indices_array = np.array(indices_array).transpose(1, 0).reshape(-1)
        else:
            indices_array = np.concatenate(indices_array[:])

        # add extra samples to make it evenly divisible
        diff_size = int(self.total_dist_size - self.total_sampled_size)
        if diff_size > 0:
            extended_indices_dist = np.append(indices_array, indices_array[:diff_size])
        else:
            extended_indices_dist = indices_array
        assert extended_indices_dist.size == self.total_dist_size

        # subsample
        offset = self.num_dist_samples * self.rank
        rank_indices = extended_indices_dist[offset: offset + self.num_dist_samples]
        assert rank_indices.size == self.num_dist_samples

        return iter(rank_indices)


def create_dataset(opt):
    logger = logging.getLogger(__name__)

    dataset = find_dataset_lib(opt.dataset)()
    dataset.initialize(opt)
    logger.info("%s is created." % opt.dataset)
    return dataset


def create_multiple_dataset(opt):
    all_datasets = []
    dataset_indices_lists = []
    indices_len = []
    for name in opt.dataset_list:
        dataset = find_dataset_lib(opt.dataset)(opt, name)
        #dataset.initialize(opt, name)
        logger.info("%s : %s is loaded, the data size is %d" % (opt.phase, name, len(dataset)))
        all_datasets.append(dataset)
        assert dataset.curriculum_list is not None, "Curriculum is None!!!"
        dataset_indices_lists.append(dataset.curriculum_list)
        indices_len.append(len(dataset.curriculum_list))
        assert len(dataset.curriculum_list) == dataset.data_size, "Curriculum list size not equal the data size!!!"
    max_len = np.max(indices_len)
    # if 'train' in opt.phase:
    #     extended_indices_list = [i * (max_len // len(i)) + list(np.random.choice(i, max_len % len(i), replace=False)) for i in dataset_indices_lists]
    #     #extended_indices_list = [i + list(np.random.choice(i, max_len-len(i))) for i in dataset_indices_lists]
    # else:
    #     extended_indices_list = dataset_indices_lists
    logger.info("%s are merged!" % opt.dataset_list)
    return all_datasets#, extended_indices_list


def find_dataset_lib(dataset_name):
    """
    Give the option --dataset [datasetname], import "data/datasetname_dataset.py"
    :param dataset_name: --dataset
    :return: "data/datasetname_dataset.py"
    """
    logger = logging.getLogger(__name__)
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls
    if dataset is None:
        logger.info("In %s.py, there should be a class name that matches %s in lowercase." % (
        dataset_filename, target_dataset_name))
        exit(0)
    return dataset

