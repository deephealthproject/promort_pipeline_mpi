# Copyright (c) 2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pyeddl.eddl as eddl
###
def init_top_layers(net):
    lays = net.layers
    for i, l in enumerate(lays):
        if l.name == 'top':
            for ltop in lays[i:]:
                ltop.initialize()
            break

### VGG16

def VGG16_tumor(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None):
    in_  = eddl.Input(in_shape)
    x = eddl.ReLu(init(eddl.Conv(in_, 64, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 64, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.Dense(x, 256)
    if dropout:
        x = eddl.Dropout(x, dropout, iw=False)
    if l2_reg:
        x = eddl.L2(x, l2_reg)
    x = eddl.ReLu(init(x,seed))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    
    net = eddl.Model([in_], [x])

    return net

def VGG16_gleason(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None):
    in_ = eddl.Input(in_shape)
    x = eddl.ReLu(init(eddl.Conv(in_, 64, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 64, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.Dense(x, 2048)
    if dropout:
        x = eddl.Dropout(x, dropout, iw=False)
    if l2_reg:
        x = eddl.L2(x, l2_reg)
    x = eddl.ReLu(init(x,seed))
    x = eddl.Dense(x, 1024)
    if dropout:
        x = eddl.Dropout(x, dropout, iw=False)
    if l2_reg:
        x = eddl.L2(x, l2_reg)
    x = eddl.ReLu(init(x,seed))
    x = eddl.Softmax(eddl.Dense(x, num_classes))

    net = eddl.Model([in_], [x])

    return net


def VGG16(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None):
    in_ = eddl.Input(in_shape)
    x = eddl.ReLu(init(eddl.Conv(in_, 64, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 64, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.Dense(x, 4096)
    if dropout:
        x = eddl.Dropout(x, dropout, iw=False)
    if l2_reg:
        x = eddl.L2(x, l2_reg)
    x = eddl.ReLu(init(x,seed))
    x = eddl.Dense(x, 4096)
    if dropout:
        x = eddl.Dropout(x, dropout, iw=False)
    if l2_reg:
        x = eddl.L2(x, l2_reg)
    x = eddl.ReLu(init(x,seed))
    x = eddl.Softmax(eddl.Dense(x, num_classes))

    net = eddl.Model([in_], [x])

    return net

def VGG16_GAP(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None):
    in_ = eddl.Input(in_shape)
    x = eddl.ReLu(init(eddl.Conv(in_, 64, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 64, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 128, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 256, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.MaxPool(eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed)), [2, 2], [2, 2])
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.ReLu(init(eddl.Conv(x, 512, [3, 3]), seed))
    x = eddl.GlobalAveragePool(x)
    x = eddl.Reshape(x, [-1])
    x = eddl.Softmax(eddl.Dense(x, num_classes))

    net = eddl.Model([in_], [x])

    return net

#### Resnet50 from scratch 

def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
  """A residual block.
  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.
  Returns:
    Output tensor for the residual block.
  """
  if conv_shortcut:
    shortcut = eddl.Conv2D(x, 4 * filters, [1,1], [stride, stride], padding='valid', name=name + '_0_conv')
    shortcut = eddl.BatchNormalization(shortcut, epsilon=1.001e-5, name=name + '_0_bn', affine=True)
  else:
    shortcut = x

  x = eddl.Conv2D(x, filters, [1, 1], [stride, stride], padding='valid', name=name + '_1_conv')
  x = eddl.BatchNormalization(x, epsilon=1.001e-5, name=name + '_1_bn', affine=True)
  x = eddl.ReLu(x,  name=name + '_1_relu')

  x = eddl.Conv2D(x, filters, [kernel_size, kernel_size], padding='same', name=name + '_2_conv')
  x = eddl.BatchNormalization(x, epsilon=1.001e-5, name=name + '_2_bn', affine=True)
  x = eddl.ReLu(x, name=name + '_2_relu')

  x = eddl.Conv2D(x, 4 * filters, [1,1], padding='valid', name=name + '_3_conv')
  x = eddl.BatchNormalization(x, epsilon=1.001e-5, name=name + '_3_bn', affine=True)

  x = eddl.Add(shortcut, x)
  x = eddl.ReLu(x, name=name + '_out')
  return x


def stack1(x, filters, blocks, stride1=2, name=None):
  """A set of stacked residual blocks.
  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.
  Returns:
    Output tensor for the stacked blocks.
  """
  x = block1(x, filters, stride=stride1, name=name + '_block1')
  for i in range(2, blocks + 1):
    x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
  return x


def ResNet50(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None):
    in_ = eddl.Input(in_shape)
    x = eddl.Pad(in_, [3, 3, 3, 3])
    x = eddl.ReLu(eddl.BatchNormalization(eddl.Conv(x, 64, [7, 7], [2, 2], "valid", False), True))
    x = eddl.Pad(x, [1, 1, 1, 1])
    x = eddl.MaxPool(x, [3, 3], [2, 2], "valid")
  
    x = stack1(x, 64, 3, stride1=1, name='conv2')
    x = stack1(x, 128, 4, name='conv3')
    x = stack1(x, 256, 6, name='conv4')
    x = stack1(x, 512, 3, name='conv5')
    
    x = eddl.GlobalAveragePool(x)
    x = eddl.Reshape(x, [-1])
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    
    net = eddl.Model([in_], [x])

    return net

### Preinitialized ONNX Networks with Imagenet. Preprocessing is the same used on pytorch vision models

def ResNet18_onnx(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None, top=False):
    resnet = eddl.download_resnet18(top=top, input_shape=in_shape) # Remove the Dense layer 
    in_ = resnet.layers[0]
    x = resnet.layers[-1]
    
    if top:
        if dropout:
            x = eddl.Dropout(x, dropout, iw=False)
        if l2_reg:
            x = eddl.L2(x, l2_reg) 
        x = eddl.Dense(x, num_classes)

    out = eddl.Softmax(x)
    
    net = eddl.Model([in_], [out])
    
    ## If the top of the pretrained network is replaced, initialize it
    if top:
        init_top_layers(net)
    
    return net

def ResNet34_onnx(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None, top=False):
    resnet = eddl.download_resnet34(top=top, input_shape=in_shape) # Remove the Dense layer 
    in_ = resnet.layers[0]
    x = resnet.layers[-1]
    
    if top:
        if dropout:
            x = eddl.Dropout(x, dropout, iw=False)
        if l2_reg:
            x = eddl.L2(x, l2_reg)
        x = eddl.Dense(x, num_classes)
        
    out = eddl.Softmax(x)
    
    net = eddl.Model([in_], [out])
    
    ## If the top of the pretrained network is replaced, initialize it
    if top:
        init_top_layers(net)
    
    return net

def ResNet50_onnx(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None, top=False):
    resnet = eddl.download_resnet50(top=top, input_shape=in_shape) # Remove the Dense layer 
    in_ = resnet.layers[0]
    x = resnet.layers[-1]
    
    if top:
        if dropout:
            x = eddl.Dropout(x, dropout, iw=False)
        if l2_reg:
            x = eddl.L2(x, l2_reg)
        x = eddl.Dense(x, num_classes)
        
    out = eddl.Softmax(x)
    
    net = eddl.Model([in_], [out])
    
    ## If the top of the pretrained network is replaced, initialize it
    if top:
        init_top_layers(net)
    
    return net


### ONNX Preinitialized VGG16 with Imagenet

def VGG16_onnx(in_shape, num_classes, seed=1234, init=eddl.HeNormal, l2_reg=None, dropout=None, top=False):
    vgg16 = eddl.download_vgg16(top=top, input_shape=in_shape) # top=True remove the imagenet trained top layer
    in_ = vgg16.layers[0]
    x = vgg16.layers[-1]
    
    if top:
        x = eddl.Dense(x, 4096)
        if dropout:
            x = eddl.Dropout(x, dropout, iw=False)
        if l2_reg:
            x = eddl.L2(x, l2_reg)
        x = eddl.ReLu(init(x,seed))
        x = eddl.Dense(x, 4096)
        if dropout:
            x = eddl.Dropout(x, dropout, iw=False)
        if l2_reg:
            x = eddl.L2(x, l2_reg)
        x = eddl.ReLu(init(x,seed))
        x = eddl.Dense(x, num_classes)
        
    out = eddl.Softmax(x)
    
    net = eddl.Model([in_], [out])
    
    ## If the top of the pretrained network is replaced, initialize it
    if top:
        init_top_layers(net)
        
    return net


### DNN for tissue detection

def tissue_detector_DNN():
    in_ = eddl.Input([3])

    layer = in_
    layer = eddl.ReLu(eddl.Dense(layer, 50))
    layer = eddl.ReLu(eddl.Dense(layer, 50))
    layer = eddl.ReLu(eddl.Dense(layer, 50))
    out = eddl.Softmax(eddl.Dense(layer, 2))
    net = eddl.Model([in_], [out])

    return net


### Main Function to build the preferred network
def get_net(net_name='vgg16', in_shape=[3,256,256], num_classes=2, full_mem=True, net_init='he', dropout=None, l2_reg=None):
    ### mem
    if full_mem:
        mem = 'full_mem'
    else:
        mem = 'low_mem'
    
    net_init = eddl.HeNormal

    ### Get Network
    if net_init == 'glorot':
        net_init = eddl.GlorotNormal

    ## Network definition
    build_init_weights = True
    if net_name == 'vgg16_tumor':
        net = VGG16_tumor(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout)
    elif net_name == 'vgg16_gleason':
        net = VGG16_gleason(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout)
    elif net_name == 'vgg16_gap':
        net = VGG16_GAP(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout)
    elif net_name == 'vgg16':
        net = VGG16(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout)
    elif net_name == 'resnet50':
        net = ResNet50(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout)
    elif net_name == 'resnet18_onnx':
        net = ResNet18_onnx(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout, top=True)
        build_init_weights = False
    elif net_name == 'resnet34_onnx':
        net = ResNet34_onnx(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout, top=True)
        build_init_weights = False
    elif net_name == 'resnet50_onnx':
        net = ResNet50_onnx(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout, top=True)
        build_init_weights = False
    elif net_name == 'vgg16_onnx':
        net = VGG16_onnx(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout, top=True)
        build_init_weights = False
    elif net_name == 'resnet50_onnx_imagenet':
        net = ResNet50_onnx(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout)
        build_init_weights = False
    elif net_name == 'vgg16_onnx_imagenet':
        net = VGG16_onnx(in_shape, num_classes, init=net_init, l2_reg=l2_reg, dropout=dropout)
        build_init_weights = False
    else:
        return None
    print (build_init_weights)
    
    return net, build_init_weights
