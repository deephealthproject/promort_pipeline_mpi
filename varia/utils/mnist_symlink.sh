#!/bin/bash
DEST=~/data
LINK_DEST=~/examples/mnist/cpp
ln -s $DEST/mnist_trX.bin $LINK_DEST/mnist_trX.bin
ln -s $DEST/mnist_trY.bin $LINK_DEST/mnist_trY.bin
ln -s $DEST/mnist_tsX.bin $LINK_DEST/mnist_tsX.bin
ln -s $DEST/mnist_tsY.bin $LINK_DEST/mnist_tsY.bin

LINK_DEST=~/examples/mnist/python
ln -s $DEST/mnist_trX.bin $LINK_DEST/mnist_trX.bin
ln -s $DEST/mnist_trY.bin $LINK_DEST/mnist_trY.bin
ln -s $DEST/mnist_tsX.bin $LINK_DEST/mnist_tsX.bin
ln -s $DEST/mnist_tsY.bin $LINK_DEST/mnist_tsY.bin
