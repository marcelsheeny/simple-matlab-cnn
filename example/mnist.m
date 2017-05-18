clc;
close all;
clear all;

% add external folders
addpath(genpath('../'));

% download and extract mnist dataset
if (~exist('train-images-idx3-ubyte'))
    urlwrite('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz','train-images-idx3-ubyte.gz');
    gunzip('train-images-idx3-ubyte.gz')
    delete train-images-idx3-ubyte.gz;
end
if (~exist('train-labels-idx1-ubyte'))
    urlwrite('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz','train-labels-idx1-ubyte.gz');
    gunzip('train-labels-idx1-ubyte.gz')
    delete train-labels-idx1-ubyte.gz;
end
if (~exist('t10k-images-idx3-ubyte'))
    urlwrite('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz','t10k-images-idx3-ubyte.gz');
    gunzip('t10k-images-idx3-ubyte.gz')
    delete t10k-images-idx3-ubyte.gz;
end
if (~exist('t10k-labels-idx1-ubyte'))
    urlwrite('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz','t10k-labels-idx1-ubyte.gz');
    gunzip('t10k-labels-idx1-ubyte.gz')
    delete t10k-labels-idx1-ubyte.gz;
end

% using helper function from http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

im = reshape(images(:,1), 28, 28 );

model = network();
model.add_layer('type','input', 'size', [28, 28, 1]);
model.add_layer('type','conv','filters',16,'size',[3,3]);
model.add_layer('type','relu');
model.add_layer('type','max_pool','size',[2,2]);
model.add_layer('type','max_pool','size',[2,2]);
model.add_layer('type','dense','size',128);
model.add_layer('type','relu');
model.add_layer('type','dense','size',10);
model.add_layer('type','softmax');

image = model.forward(im);

