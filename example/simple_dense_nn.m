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
classes = [zeros(length(data1),2); zeros(length(data2),2)];
classes(1:length(data1),1) = 1;
classes(length(data1):end,2) = 1;

% randomize indexes
randind = randperm(size(data,1));
data = data(randind,:);
classes = classes(randind,:);


% normalize
data = (data - min(data(:))) / (max(data(:)) - min(data(:)));

train_data = [data(1:1400,:)];
train_labels = [classes(1:1400,:)];

test_data = [data(1401:2000,:)];
test_labels = [classes(1401:2000,:)];


% init network
model = neural_network();
model.add_layer('type','input', 'size', [2, 1, 1]);
model.add_layer('type','dense','size',5);
model.add_layer('type','relu');
model.add_layer('type','dense','size',5);
model.add_layer('type','relu');
model.add_layer('type','dense','size',2);
model.add_layer('type','relu');
model.add_layer('type','softmax');

%a = model.forward([0 0]);

epochs = 1000;
batch = 30;
val_percentage = 0.2;
epoch_val = 50;
l_r = 0.001;

[err_train, err_val] = model.train(epochs, l_r, batch, train_data, train_labels, val_percentage, epoch_val);

[results, acc] = model.test(test_data, test_labels);

%draw
plot(test_data(test_labels==1,1), test_data(test_labels==1,2), '*'); hold on;
plot(test_data(test_labels==2,1), test_data(test_labels==2,2), '*')



