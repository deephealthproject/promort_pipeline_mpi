#!/bin/bash
DEST=~/data
mkdir -p $DEST && cd $DEST
wget -q --show-progress https://www.dropbox.com/s/khrb3th2z6owd9t/mnist_trX.bin
wget -q --show-progress https://www.dropbox.com/s/m82hmmrg46kcugp/mnist_trY.bin
wget -q --show-progress https://www.dropbox.com/s/7psutd4m4wna2d5/mnist_tsX.bin
wget -q --show-progress https://www.dropbox.com/s/q0tnbjvaenb4tjs/mnist_tsY.bin
