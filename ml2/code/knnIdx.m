function [k_nearest_index]=knnIdx(sample_index,k,distance_matrix)  

[~,index]=sort(distance_matrix(:,sample_index));  
k_nearest_index=index(2:k+1,1);

end  
