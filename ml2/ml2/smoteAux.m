function Xs_new  = smoteAux(  Xs, N, K )
%SMOTEAUX �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��


if(N<=1) 
    Xs_new = Xs;
else
    [sample_num, attr_num] = size(Xs);
    new_sample_num = N*sample_num;
    Xs_new = zeros(new_sample_num, attr_num);
    new_index = 1;
    
    for i = 1 : sample_num
        knnidx = knnsearch(Xs, Xs, 'K', K);
        for j = 1 : N
            nnrand = randperm(K, 1);
            nn_idx = knnidx(i, nnrand);
            for k = 1 : attr_num
                diff = Xs(nn_idx, k) - Xs(i, k);
                gap = rand(1, 1);
                Xs_new(new_index, k) = Xs(i, k) + gap*diff;
            end
            new_index = new_index+1;
        end
    end
    Xs_new = [Xs;Xs_new];
    
end

end
