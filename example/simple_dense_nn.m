clc;
close all;
clear all;

% add external folders
addpath(genpath('../'));

% gen data
% gen class 1
mu = [0,0];
sigma = [1,0;0,1];
samples = 1000;
data1 = mvnrnd(mu,sigma,samples);

% gen class 2
mu = [5,5];
sigma = [1,0;0,1];
samples = 1000;
data2 = mvnrnd(mu,sigma,samples);

data = [data1; data2];
%classes = [1*ones(length(data1),1); 2*ones(length(data2),1)];

% normalize
data = (data - min(data(:))) / (max(data(:)) - min(data(:)));

train_data = [data1(1:700,:);data2(1:700,:)];
train_labels = [1*ones(700,1); 2*ones(700,1)];

test_data = [data1(701:1000,:);data2(701:1000,:)];
test_labels = [1*ones(300,1); 2*ones(300,1)];


% init network
model = neural_network();
model.add_layer('type','input', 'size', [2, 1, 1]);
model.add_layer('type','dense','size',5);
model.add_layer('type','relu');
model.add_layer('type','dense','size',5);
model.add_layer('type','relu');
model.add_layer('type','dense','size',2);
model.add_layer('type','softmax');

%a = model.forward([0 0]);

epochs = 1000;
batch = 30;
val_percentage = 0.2;
epoch_val = 50;
l_r = 0.001;

[err_train, err_val] = model.train(epochs, batch, l_r, train_data, train_labels, val_percentage, epoch_val);