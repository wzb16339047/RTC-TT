% fold matrix based on Tensor-Train
function [X] = FoldTT(X, dim, i)

X=reshape(X,dim);