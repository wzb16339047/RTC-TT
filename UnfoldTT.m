% Unfold the tensor based on Tensor-Train
function [X] = UnfoldTT( X, dim, i )
product_1=1;
product_2=1;
for j=1:i
    product_1=product_1*dim(j);
end
for k=i+1:ndims(X)
    product_2=product_2*dim(k);
end
X = reshape(X, product_1, product_2);