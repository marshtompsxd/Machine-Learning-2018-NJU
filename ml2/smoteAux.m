function Xs_new  = smoteAux(  Xs, N, K )

if(N<=1) 
    Xs_new = Xs;
else
    [sample_num, attr_num] = size(Xs);
    new_sample_num = N*sample_num;
    Xs_new = zeros(new_sample_num, attr_num);
    index_num = 1;
    distanceM=dist(Xs');
    sample_index=1:sample_num;     
    
    for i = 1 : sample_num
        knnidx = knnIdx(sample_index(i), K, distanceM);
        for j = 1 : N
            nnrand = randperm(K, 1);
            nnidx = knnidx(nnrand);
            for k = 1 : attr_num
                difference = Xs(nnidx, k) - Xs(i, k);
                deviation = rand(1, 1);
                Xs_new(index_num, k) = Xs(i, k) + deviation*difference;
            end
            index_num = index_num+1;
        end
    end
    Xs_new = [Xs;Xs_new];
    
end

end

