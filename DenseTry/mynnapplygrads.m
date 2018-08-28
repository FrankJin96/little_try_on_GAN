function nn = mynnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
        for i = 1 : 2
            if(nn.weightPenaltyL2>0)
                dW = nn.dW{i} + nn.weightPenaltyL2 * nn.W{i};
            else
                dW = nn.dW{i};
            end

            dW = nn.learningRate * dW;

            if(nn.momentum>0)
                nn.vW{i} = nn.momentum*nn.vW{i} + dW;
                dW = nn.vW{i};
            end
            if i == 1
                nn.W{i} = nn.W{i} - dW;                                             %�����������ѡһ
            end
%% yinggh,nn.StopWΪ0�򲻸��£�isStopWΪ1�����,

%%����������������������������������������������������������������������������������������������
%��������������Ϊ�����Լ��ĵĺ��������˾������˵�ʲô������û����Ӧ���������ʼ��StopW����������
%%����������������������������������������������������������������������������������������������
%         if(isfield(nn,'StopW'))
%             disp('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
%             if(nn.StopW(i)==0)
%                 nn.W{i} = nn.W{i} - dW*0;
%             elseif(nn.StopW(i)==1)
%                 nn.W{i} = nn.W{i} - dW*1;
%             else
%                 nn.W{i} = nn.W{i} - dW*1;
%             end
%         else
%             nn.W{i} = nn.W{i} - dW;
        end
end
