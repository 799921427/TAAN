from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from random import shuffle
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

class CamSampler(Sampler):
    def __init__(self, data_source, need_cam, num=0):
        self.data_source = data_source
        self.index_dic = []
        self.id_cam = [[] for _ in range(533)]
        
        if num>0:
            for index, (_, pid, cam) in enumerate(data_source):
                if index % 2 == 1: continue
                if cam in need_cam:
                    self.id_cam[pid].append(index)
            for i in range(533):
                if len(self.id_cam[i])>num:
                    self.index_dic.extend(self.id_cam[i][:num])
        else:
            for index, (_, pid, cam) in enumerate(data_source):
                if index % 2 == 1: continue
                if cam in need_cam:
                    self.index_dic.append(index)

    def __len__(self):
        return len(self.index_dic)

    def __iter__(self):
        return iter(self.index_dic)

class CamRandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=2):
        self.data_source = data_source
        self.num_instances = num_instances
        if num_instances % 2 > 0:
            raise ValueError("The num_instances should be a even number")
        self.index_dic_I = defaultdict(list)
        self.index_dic_IR = defaultdict(list)
        for index, (name, pid, cam) in enumerate(data_source):
            if cam == 2 or cam == 5:
                self.index_dic_IR[pid].append(index)
            else:
                self.index_dic_I[pid].append(index)
        self.pids = list(self.index_dic_I.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid_I = self.pids[i]
            pid_IR = self.pids[i]
            t_I = self.index_dic_I[pid_I]
            t_IR = self.index_dic_IR[pid_IR]
            if len(t_I) >= self.num_instances / 2:
                t_I = np.random.choice(t_I, size=int(self.num_instances / 2), replace=False)
            else:
                t_I = np.random.choice(t_I, size=int(self.num_instances / 2), replace=True)
            if len(t_IR) >= self.num_instances / 2:
                t_IR = np.random.choice(t_IR, size=int(self.num_instances / 2), replace=False)
            else:
                t_IR = np.random.choice(t_IR, size=int(self.num_instances / 2), replace=True)
            # ret.extend(t_I)
            # ret.extend(t_IR)
            if len(t_I) > 1 and t_I[0] == t_I[1] - 1:
                t_I = np.append(t_I, [t_I[0]], axis=0)
                t_I = np.append(t_I, [t_I[1]], axis=0)
            else:
                for j in range(self.num_instances / 2):
                    for i in range(len(t_I)):
                        # print(len(t_I))
                        if t_I[i] % 2 == 0 :
                            if i+1 == len(t_I):
                                t_I = np.append(t_I, [t_I[i] + 1], axis=0)
                                break
                            elif t_I[i+1] !=t_I[i] + 1 and i+1 < len(t_I):
                                t_I = np.insert(t_I, i+1, [t_I[i] + 1], 0)
                                break
                        elif t_I[i] % 2 != 0 :
                            if i == 0:
                                t_I = np.insert(t_I, i, [t_I[i] - 1], 0)
                                break
                            elif t_I[i-1] != t_I[i] - 1 and i-1 >=0 :
                                t_I = np.insert(t_I, i, [t_I[i]-1], 0)
                                break

            if len(t_IR) > 1 and t_IR[0] == t_IR[1] - 1:
                t_IR = np.append(t_IR, [t_IR[0]], axis=0)
                t_IR = np.append(t_IR, [t_IR[1]], axis=0)
            else:
                for j in range(self.num_instances / 2):
                    # print("j:%d" %(j))
                    for i in range(len(t_IR)):
                        # print(len(t_IR))
                        # print(t_IR)
                        if t_IR[i] % 2 == 0:
                            if i + 1 == len(t_IR):
                                t_IR = np.append(t_IR, [t_IR[i] + 1], axis=0)
                                break
                            elif t_IR[i + 1] != t_IR[i] + 1 and i + 1 < len(t_IR):
                                t_IR = np.insert(t_IR, i + 1, [t_IR[i] + 1], 0)
                                break
                        elif t_IR[i] % 2 != 0:
                            if i == 0:
                                t_IR = np.insert(t_IR, i, [t_IR[i] - 1], 0)
                                break
                            elif t_IR[i - 1] != t_IR[i] - 1 and i - 1 >= 0:
                                t_IR = np.insert(t_IR, i, [t_IR[i] - 1], 0)
                                break
            j = 0
            for i in range(self.num_instances / 2):
                # print(t_I[j])
                # print(t_I[j+1])
                # print(t_IR[j])
                # print(t_IR[j+1])

                ret.append(t_I[j])
                ret.append(t_I[j + 1])
                ret.append(t_IR[j])
                ret.append(t_IR[j + 1])
                j = j + 2

        return iter(ret)


class IDRandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=2):
        self.data_source = data_source
        self.num_instances = num_instances
        if num_instances % 2 > 0:
            raise ValueError("The num_instances should be a even number")
        self.index_dic_I = defaultdict(list)
        for index, (name, pid, cam) in enumerate(data_source):
            self.index_dic_I[pid].append(index)
        self.pids = list(self.index_dic_I.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid_I = self.pids[i]
            t_I = self.index_dic_I[pid_I]
            if len(t_I) >= self.num_instances / 2:
                t_I = np.random.choice(t_I, size=int(self.num_instances / 2), replace=False)
            else:
                t_I = np.random.choice(t_I, size=int(self.num_instances / 2), replace=True)

            # ret.extend(t_I)
            # ret.extend(t_IR)
            for j in range(self.num_instances // 2):
                ret.append(t_I[j])
        return iter(ret)