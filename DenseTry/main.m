clc
clear
%% ������ʵѵ������ 60000������ 1*784ά��28*28չ����
load mnist_uint8;                                                                               %������д���ֿ�

train_x = double(train_x(1:60000,:)) / 255;                                                     %��ѵ����X��Ϊdouble�ͣ�������255��ΪʲôҪ����255������
% ��ʵ������ΪΪ��ǩ [1 0]�� ��������Ϊ[0 1];                                                    ����ǩ��ʲô����ʲô������ʾ������yô������
train_y = double(ones(size(train_x,1),1));                                                      %train_y�ĳ�ʼ������train_x�Ĵ�Сһ�£���Ԫ��ȫΪ1
% normalize
train_x = mapminmax(train_x, 0, 1);                                                             %mapminmax����ʵ�־���Ĺ�һ����Ŀ����ʹ��������ٶȡ������������λָ�����

rand('state',0)                                                                                 %�趨�������ģʽΪstate
%% ����ģ��ѵ������ 60000������ 1*100ά
test_x = normrnd(0,1,[60000,100]); % 0-255������                                                %Ϊʲô��0-255����������
test_x = mapminmax(test_x, 0, 1);                                                               %��һ��

test_y = double(zeros(size(test_x,1),1));                                                       %��ʼ��test_y
test_y_rel = double(ones(size(test_x,1),1));                                                    %test_y_rel��ʲô�������Լ�y�ķ�����

%%
nn_G_t = nnsetup([100 784]);                                                                    %nnsetup��ʾ����һ��������ṹ��ʾ�������磬Ϊʲô�ṹ�������ģ���t��ʾʲô���壿��100��ʲô����
nn_G_t.activation_function = 'sigm';
nn_G_t.output = 'sigm';

nn_D = nnsetup([784 100 1]);
nn_D.weightPenaltyL2 = 1e-4;                            %  L2 weight decay
nn_D.dropoutFraction = 0.5;                             %  Dropout fraction   
nn_D.learningRate = 0.01;                               %  Sigm require a lower learning rate
nn_D.activation_function = 'sigm';
nn_D.output = 'sigm';
% nn_D.weightPenaltyL2 = 1e-4;  %  L2 weight decay

nn_G = nnsetup([100 784 100 1]);
nn_G.weightPenaltyL2 = 1e-4;  %  L2 weight decay
nn_G.dropoutFraction = 0.5;   %  Dropout fraction 
nn_G.learningRate = 0.01;     %  Sigm require a lower learning rate
nn_G.activation_function = 'sigm';
nn_G.output = 'sigm';
% nn_G.weightPenaltyL2 = 1e-4;  %  L2 weight decay

opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples
%%
num = 1000;
tic
for each = 1:1500
    %----------����G�������������------------------- 
    for i = 1:length(nn_G_t.W)   %�����������
        nn_G_t.W{i} = nn_G.W{i};
    end
    G_output = nn_G_out(nn_G_t, test_x);
    %-----------ѵ��D------------------------------
    index = randperm(60000);
    train_data_D = [train_x(index(1:num),:);G_output(index(1:num),:)];
    train_y_D = [train_y(index(1:num),:);test_y(index(1:num),:)];
    nn_D = nntrain(nn_D, train_data_D, train_y_D, opts);%ѵ��D
    %-----------ѵ��G-------------------------------
    for i = 1:length(nn_D.W)  %����ѵ����D���������
        nn_G.W{length(nn_G.W)-i+1} = nn_D.W{length(nn_D.W)-i+1};
    end
    %ѵ��G����ʱ��������ǩΪ1����Ϊ��������
    nn_G = nntrain(nn_G, test_x(index(1:num),:), test_y_rel(index(1:num),:), opts);
end
toc
for i = 1:length(nn_G_t.W)
    nn_G_t.W{i} = nn_G.W{i};
end
fin_output = nn_G_out(nn_G_t, test_x);
matrixilize;

