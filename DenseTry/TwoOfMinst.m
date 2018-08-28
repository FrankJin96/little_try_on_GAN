clear;
clc;
load mnist_uint8;
train1_x = zeros(10,784);
k = 0;
for i = 1:size(train_x,1)
    if train_y(i,:) == [0 0 0 0 0 0 0 0 1 0]
        k = k + 1;
        train1_x(k,:) = train_x(i,:);
    end
end
train1_x = double(train1_x(1:size(train1_x,1),:)) / 255;
train1_y = ones(size(train1_x,1),1);
% save 2inmnist_doub;