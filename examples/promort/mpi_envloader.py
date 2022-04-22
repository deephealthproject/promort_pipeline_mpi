from cassandradl import CassandraDataset
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from pathlib import Path

import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

import models

from OPT_MPI import mpi_env, sgd_mpi
import subprocess
import models

class EnvLoader():
    def __init__(self, inet_pass, n_sync, val_ratio, test_ratio, augs_on, batch_size,
                 max_patches, cass_row_fn, cass_datatable,
                 net_name, size, num_classes, label, lr, gpus, net_init,
                 dropout, l2_reg, init_weights_fn, seed):
        
        self.MP = mpi_env(n_sync, bl=512)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num = self.MP.mpi_size
        self.inet_pass = inet_pass
        self.cd = None
        self.label = label
        self.cass_row_fn = cass_row_fn
        self.cass_datatable = cass_datatable
        self.max_patches = max_patches
        self.batch_size = batch_size
        self.augs_on = augs_on
        self.seed = seed
        self.net_name = net_name
        self.size = size
        self.num_classes = num_classes
        self.lr = lr
        self.gpus = gpus
        self.net_init = net_init
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.net = None
        self.ngpus = 0
        self.init_weights_fn = init_weights_fn

        if gpus:
            # Discover how many gpus can be used
            result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
            self.ngpus = len(result.stdout.decode('utf-8').split('\n'))-1

    def split_setup(self, seed):
        self.seed = seed
        tr_ratio = [(1 - self.val_ratio - self.test_ratio) / self.num] * self.num
        val_ratio = [self.val_ratio / self.num] * self.num 
        test_ratio = [self.test_ratio / self.num] * self.num 
        split_ratios = tr_ratio + val_ratio + test_ratio
        self.cd.split_setup(split_ratios=split_ratios,
                            max_patches=self.max_patches, seed=seed)
                            #max_patches=8000, seed=seed)
        self.cd.set_batchsize(bs=self.batch_size, full_batches=True)
        
        if self.MP.mpi_rank == 0:
            print ("SPLIT RATIOS: %r" % split_ratios)
            self.cd.save_splits('/tmp/curr_split.pckl')
    
    def cassandra_setup(self, seed):
        print('####### Starting Cassandra dataset')
        print('Seed = %r ' % seed)
        ap = PlainTextAuthProvider(username='prom',
                                   password=self.inet_pass)
        cd = CassandraDataset(ap, ['cassandra-db'], seed=seed)
        cd.load_rows(self.cass_row_fn)
        cd.label_col = self.label
        cd.init_datatable(table=self.cass_datatable)
        self.cd = cd
        self.split_setup(seed)

    def start(self, seed):
        # cassandra
        if (self.cd is None):  # first execution, create data structure
            self.cassandra_setup(seed)
        if (self.seed != seed):  # seed changed, reshuffle all
            print ("Split setup")
            self.split_setup(seed)
        # eddl network
        if (self.net is None):
            self.net_setup()

    def net_setup(self):
        in_shape = [3, self.size[0], self.size[1]]
        # set net init
        if (self.net_init=='HeNormal'):
            net_init = eddl.HeNormal
        elif (self.net_init=='GlorotNormal'):
            net_init = eddl.GlorotNormal
        else:
            raise ValueError('net_init can only be HeNormal or GlorotNormal')
        
        # set network
        self.net, build_init_weights = models.get_net(net_name=self.net_name, in_shape=in_shape, num_classes=self.num_classes, full_mem=True, net_init=self.net_init, dropout=self.dropout, l2_reg=self.l2_reg)
  
        # Build network
        if self.num == 1:
            ## Single node execution
                
            eddl.build(
                self.net,
                eddl.sgd(self.lr),
                ["soft_cross_entropy"],
                ["categorical_accuracy"],
                eddl.CS_GPU(self.gpus, mem="full_mem", lsb=1) if self.gpus else eddl.CS_CPU(),
                init_weights = build_init_weights
                )

        else:
            ## multinode and/or multigpu
            if self.gpus:
                gpu_mask = [0] * self.ngpus
                gpu_mask[self.MP.mpi_rank % self.ngpus] = 1
            else:
                gpu_mask = []

            eddl.build(
                self.net,
                sgd_mpi(self.MP, self.lr),
                ["soft_cross_entropy"],
                ["categorical_accuracy"],
                eddl.CS_GPU(gpu_mask, mem="full_mem", lsb=1) if self.gpus else eddl.CS_CPU(),
                init_weights = build_init_weights
                )

        eddl.summary(self.net)
   
        ## Load weights if requested
        if self.init_weights_fn and '_onnx' not in self.net_name:
            print ("Loading initialization weights: %s" % self.init_weights_fn)
            eddl.load(self.net, self.init_weights_fn)
    
