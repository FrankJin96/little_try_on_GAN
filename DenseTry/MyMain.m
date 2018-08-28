clc;
clear
A = clock;
global mse;
MSE = zeros(10,1);
%% 构造真实训练样本 60000个样本 1*784维（28*28展开）
load mnist_uint8;                                       %载入手写体字库

train_x = double(train_x(1:60000,:)) / 255;             %将训练集X化为double型，并除以255（为什么要除以255？？）
% 真实样本认为为标签 [1 0]； 生成样本为[0 1];             [标签是什么，用什么变量表示？？是y么？？]
train_y = double(ones(size(train_x,1),1));              %train_y的初始化，与train_x的大小一致，且元素全为1
% normalize
train_x = mapminmax(train_x, 0, 1);                     %mapminmax函数实现矩阵的归一化，目的是使提高收敛速度。数据不需要恢复，归一化之后图片的形状不改变

rand('state',0)                                         %设定随机数的模式为state
%% 构造模拟训练样本 60000个样本 1*100维
test_x = normrnd(0,1,[60000,100]); % 0-255的整数        %为什么是0-255的整数？？
test_x = mapminmax(test_x, 0, 1);                       %归一化

test_y = double(zeros(size(test_x,1),1));               %初始化test_y
test_y_rel = double(ones(size(test_x,1),1));            %测试集y的反

%%
nn_G_t = nnsetup([100 361 784]);                            %nnsetup表示建立一个如输入结构所示的神经网络，100是因为test的大小为100，train的大小为784，t表示完全的生成器
nn_G_t.activation_function = 'sigm';                    %设置激励函数为sigm，输出函数为sigm；
nn_G_t.output = 'sigm';

nn_D = nnsetup([784 100 1]);
nn_D.weightPenaltyL2 = 1e-4;                            %  L2 weight decay
nn_D.dropoutFraction = 0.5;                             %  Dropout fraction
nn_D.learningRate = 0.01;                               %  Sigm require a lower learning rate
nn_D.activation_function = 'sigm';
nn_D.output = 'sigm';

nn_G = nnsetup([100 361 784 100 1]);
nn_G.weightPenaltyL2 = 1e-4;                            %  L2 weight decay
nn_G.dropoutFraction = 0.5;                             %  Dropout fraction
nn_G.learningRate = 0.01;                               %  Sigm require a lower learning rate
nn_G.activation_function = 'sigm';
nn_G.output = 'sigm';

opts.numepochs =  1;                                    %  Number of full sweeps through data
opts.batchsize = 50;                                    %  Take a mean gradient step over this many samples
%%
num = 1000;
for each = 1:50
    %----------计算G的输出：假样本------------------
    for i = 1:length(nn_G_t.W)   %共享网络参数
        nn_G_t.W{i} = nn_G.W{i};
    end
    G_output = nn_G_out(nn_G_t, test_x);
    %-----------训练D------------------------------
    for iter = 1 : 2
        index = randperm(60000);
        train_data_D = [train_x(index(1:num),:);G_output(index(1:num),:)];
        train_y_D = [train_y(index(1:num),:);test_y(index(1:num),:)];
        nn_D = nntrain(nn_D, train_data_D, train_y_D, opts);%训练D
    end
    %-----------训练G-------------------------------
    for i = 1:length(nn_D.W)  %共享训练的D的网络参数
        nn_G.W{length(nn_G.W)-i+1} = nn_D.W{length(nn_D.W)-i+1};
    end
    %训练G：此时假样本标签为1，认为是真样本
%     nn_G = nntrain(nn_G, test_x(index(1:num),:), test_y_rel(index(1:num),:), opts);
    nn_G = mynntrain(nn_G, test_x(index(1:num),:), test_y_rel(index(1:num),:), opts);
    MSE(each,1) = mse;
    if rem(each,50) == 0
        disp(each);
    end
end
for i = 1:length(nn_G_t.W)
    nn_G_t.W{i} = nn_G.W{i};
end
fin_output = nn_G_out(nn_G_t, test_x);
matrixilize;

A
clock
% disp('MyMain has ended...')
% disp('Plz check the result!')
% load chirp;
% sound(y,Fs);