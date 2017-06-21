clc;
close all;
clear all;

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
classes = [1*ones(length(data1),1); 2*ones(length(data2),1)];

data_n = (data - min(data(:))) / (max(data(:)) - min(data(:)));

%init weights
% 2 input layers
% 4 hidden layers
% 4 hidden layers
% 2 output layers

W1 = rand(4, 2);
W2 = rand(4, 4);
W3 = rand(2, 4);

% generate all possible inputs
xrange = [0 1];
yrange = [0 1];
% step size for how finely you want to visualize the decision boundary.
inc = 0.05;
 
% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
  
% size of the (x, y) image, which will also be the size of the 
% decision boundary image that is used as the plot background.
% image_size = size(x);
%  
xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.

labels = [];

for i=1:length(xy)
   I = xy(i,:)';
   
   o1 = W1*I;
   % sigmoid
   o1 = sigmf(o1,[1 0]);
   o2 = W2*o1;
   
   % sigmoid
   o2 = sigmf(o2,[1 0]);
   o3 = W3*o2;
   
   % sigmoid
   o3 = sigmf(o3,[1 0]);
   
   %softmax
   s = exp(o3)/sum(exp(o3));
   [~, idx] = max(s);
   labels(end+1,1) = idx;
   xy(i,:)'
   s
   idx
end

image_size = size(x);

decisionmap = reshape(labels, image_size);
%  
% %show the image
imagesc(xrange,yrange,decisionmap);
% hold on;
set(gca,'ydir','normal');
%  
% % colormap for the classes:
% % class 1 = light red, 2 = light green, 3 = light blue
cmap = hsv(max(classes));
colormap(cmap); hold on;

gscatter(data_n(:,1), data_n(:,2), classes,'br','xo');