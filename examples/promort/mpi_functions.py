from cassandradl import CassandraDataset
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from pathlib import Path

import time
from tqdm import trange, tqdm
import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import random
import pickle, os
import datetime

def train(el, epochs, lr, gpus, dropout, l2_reg, seed, out_dir, save_weights, find_opt_lr=False):
    
    MP = el.MP
    rank = MP.mpi_rank
    mpi_size = MP.mpi_size

    print('Starting train function')
    t0 = time.time()

    # Get Environment
    el.start(seed)
    cd = el.cd
    net = el.net
    out = net.layers[-1]

    t1 = time.time()
    print("Time to load the Environment  %.3f" % (t1-t0))
    t0 = t1

    ###################
    ## Training step ##
    ###################
    
    print("Starting training", flush=True)

    if rank == 0: # Only task 0 takes account of whole stats
        loss_l = []
        acc_l = []
        val_loss_l = []
        val_acc_l = []
        ts_l = []
        val_ts_l = []
            
        patience_cnt = 0
        val_acc_max = 0.0

    acc_fn = eddl.getMetric("categorical_accuracy")
    loss_fn = eddl.getLoss("soft_cross_entropy")
    
    ### Main loop across epochs
    t0 = time.time()
    
    ## Make the split setup before the main loop. Then mix only training splits
    #el.split_setup(seed)
    train_splits_l = [iii for iii in range(0, el.num)]
    val_splits_l = [iii for iii in range(el.num, 2*el.num)]
    tr_num_batches = min(cd.num_batches[0:el.num]) - 1 # FIXME: Using the minimum among all batches not the local ones
    val_num_batches = min(cd.num_batches[el.num:el.num*2]) - 1 # FIXME: Using the minimum among all batches not the local ones
    
    print ("NUM BATCHES %r" % cd.num_batches)
    print(tr_num_batches)
    print(val_num_batches)
    
    start_time = datetime.datetime.now()


    #### Code used to find best learning rate. Comment it to perform an actual training
    if find_opt_lr:
        max_epochs = epochs
        lr_start = lr
        lr_end = lr * 1.0e4
        lr_f = lambda x: 10**(np.log10(lr_start) + ((np.log10(lr_end)-np.log10(lr_start))/max_epochs)*x)
    ####

    for e in range(epochs):
        ####
        ### Training 
        ####

        ## SET LT
        if find_opt_lr:
            print (f"SETTING A NEW LR: {lr_f(e)}")
            eddl.setlr(net, [lr_f(e)])
          
        if rank == 0:
            print ("\n\n\n*** Training ***")
            epoch_acc_l = []
            epoch_loss_l = []
            epoch_val_acc_l = []
            epoch_val_loss_l = []

        ### Recreate splits to shuffle among workers but with the same seed to get same splits
        eddl.reset_loss(net)
        #seed = random.getrandbits(32)
        
        cd.mix_splits(train_splits_l)
        
        #pbar = tqdm(range(tr_num_batches))
        pbar = tqdm(range(tr_num_batches * mpi_size))

        for b_index in range(tr_num_batches):
            # Init local weights to a zero structure equal in size and shape to the global one
            t0 = time.time()
            split_index = rank # Local split. If ltr == 1 --> split_index = rank
                
            x, y = cd.load_batch(split_index)

            x.div_(255.0)
            tx, ty = [x], [y]
                    
            #print (f'Train batch rank: {rank}, ep: {e}, macro_batch: {mb}, local training rank: {lt}, inidipendent iteration: {s_it}') 
            eddl.train_batch(net, tx, ty)
            
            net_out = eddl.getOutput(net.layers[-1]).getdata() 
            loss = eddl.get_losses(net)[0]
            acc = eddl.get_metrics(net)[0]
        
            # Loss and accuracy synchronization among ranks
            loss = MP.Gather_and_average(loss)
            acc = MP.Gather_and_average(acc)
            
            if rank == 0:
                msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, epochs, b_index, tr_num_batches + 1, loss, acc)
                pbar.set_postfix_str(msg)
                epoch_loss_l.append(loss)
                epoch_acc_l.append(acc)
         
            pbar.update(mpi_size)

        ## End of macro batches
        pbar.close()
        
        # Store train epoch datetime
        train_datetime = datetime.datetime.now()

        ######
        ## Validation 
        ######
        if rank == 0:
            print ("\n\n\n*** Evaluation ***")

        eddl.reset_loss(net)
        seed = random.getrandbits(32)
        for sp in val_splits_l:
            cd.rewind_splits(sp)
        
        epoch_val_acc_l = []
        epoch_val_loss_l = []
        
        #pbar = tqdm(range(val_num_batches))
        pbar = tqdm(range(val_num_batches * mpi_size))
        
        #for b_index, mb in enumerate(pbar):
        for b_index in range(val_num_batches):
            # Init local weights to a zero structure equal in size and shape to the global one
            t0 = time.time()
            split_index = el.num + rank # Local validation split
                
            x, y = cd.load_batch(split_index)
            x.div_(255.0)
            tx, ty = [x], [y]
                    
            #print (f'Train batch rank: {rank}, ep: {e}, macro_batch: {mb}, local training rank: {lt}, inidipendent iteration: {s_it}') 
            eddl.forward(net, tx)
            
            net_out = eddl.getOutput(net.layers[-1]) 
       
            sum_ca = 0.0 ## sum of samples accuracy within a batch
            sum_ce = 0.0 ## sum of losses within a batch

            n = 0
            for k in range(x.getShape()[0]):
                result = net_out.select([str(k)])
                target = y.select([str(k)])
                ca = acc_fn.value(target, result)
                ce = loss_fn.value(target, result)
                sum_ca += ca
                sum_ce += ce
                n += 1
            
            loss = (sum_ce / n)
            acc = (sum_ca / n)
            
            # Loss and accuracy synchronization among ranks
            loss = MP.Gather_and_average(loss)
            acc = MP.Gather_and_average(acc)

            if rank == 0:
                # Only rank 0 print progression bar
                epoch_val_loss_l.append(loss)
                epoch_val_acc_l.append(acc)
                msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(e + 1, epochs, b_index, val_num_batches + 1, loss, acc)
                pbar.set_postfix_str(msg)
                
            pbar.update(mpi_size)

        ## End of macro batches
        pbar.close()
        
        # Store validation epoch datetime
        val_datetime = datetime.datetime.now()

        # Compute Epoch loss and acc and store history
        if rank == 0:
            loss_l.append(np.mean(epoch_loss_l))
            acc_l.append(np.mean(epoch_acc_l))
            val_loss_l.append(np.mean(epoch_val_loss_l))
            val_acc_l.append(np.mean(epoch_val_acc_l))
            ts_l.append(train_datetime)
            val_ts_l.append(val_datetime)
            if out_dir:
                history = {'loss': loss_l, 'acc': acc_l, 'val_loss': val_loss_l, 'val_acc': val_acc_l, 'start_time': start_time, 'ts':ts_l, 'val_ts': val_ts_l}
                pickle.dump(history, open(os.path.join(out_dir, 'history.pickle'), 'wb'))
                if save_weights:
                    path = os.path.join(out_dir, "weights_ep_%s_vacc_%.2f.bin" % (e, val_acc_l[-1]))
                    eddl.save(net, path, "bin")
        

    ## End of Epochs
    if rank == 0:
        return loss_l, acc_l, val_loss_l, val_acc_l, ts_l, val_ts_l, start_time
    else:
        return None, None, None, None, None, None, None
