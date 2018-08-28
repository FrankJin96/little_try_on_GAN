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
                nn.W{i} = nn.W{i} - dW;                                             %这行与下面二选一
            end
%% yinggh,nn.StopW为0则不更新，isStopW为1则更新,

%%！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
%！！！！！以下为博主自己改的函数，个人觉得少了点什么，至少没见到应该在哪里初始化StopW！！！！！
%%！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
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
