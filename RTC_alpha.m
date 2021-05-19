%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Auto-weighted Robust Low Rank Tensor Completion based on TT-rank (RTC-TT)
% Time: 14/8/2019
% Reference: "Tensor Completion for Estimating Missing Values 
% in Visual Data", PAMI, 2012.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xlast, N, Show,Observed,errList,psnrList] = RTC_alpha(T,Y, known, maxIter, epsilon,mr,noise,lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%
% min(X,N): \min_{X_{[k]},\mathcal{N}} \sum_{k=1}^{N-1} \alpha_k \lVert M_{k}\rVert_* + \lambda \lVert \mathcal{N} \rVert_1
% s.t.  \mathcal{Y}_\Omega = \mathcal{Z}_\Omega,\mathcal{Z=X+N}, M_k=X_{[k]},
%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 10
    dim = size(T);
    Omega = zeros(dim);
    Omega(known) = 1;
    Omega = logical(Omega);
    % Initializing X and Z
    Z = Y;
    Z(logical(1-Omega)) = mean(Y(Omega));
    X = Z;
    Show = Y;
    Show(logical(1-Omega)) = 0;
    Observed = Y;
    Observed(logical(1-Omega)) = mean(Y(Omega));
end

% Initializing
N=zeros(dim);
V=zeros(dim);
errList = zeros(maxIter, 1);
psnrList = zeros(maxIter, 1);
loss= zeros(maxIter, 1);
M = cell(ndims(T), 1);
Q=cell(ndims(T), 1);

alpha = ones(1, ndims(T));                      %\alpha
alpha = alpha ./ sum(alpha);

beta = (mr+0.0001)*1*ones(1, ndims(T)); 
% beta = (mr+0.0001)*0.001*ones(1, ndims(T));     %image
p=beta(1)*noise/mr;      %image
%  p = 2;
% lambda= 10; 


% lambda = 1/sqrt(256*3);
% p = noise+0.01
% beta = 0.1*ones(1,ndims(T))*(mr/10+0.1);   %video
betasum = sum(beta);
tau = alpha./ beta;                             %thresholding

mu1= p/((p+beta(1)))
mu2= 1-mu1
                                    %\lambda
gamma = 1;
normT = norm(T(:));
for i = 1:ndims(V)
        Q{i}=UnfoldTT(V,dim,i);
end

%Iterations
for k = 1:maxIter
    if mod(k, 2) == 0
        fprintf('RTC-TT: iterations = %d   difference=%f\n', k, errList(k-1));
    end
    alpha = Weights(X,gamma); %Auto-weighted
    
    alpha = alpha ./ sum(alpha);
    tau = alpha./ beta;
    Xsum = 0;
    %Update M
    for i = 1:ndims(T)
        M{i} = Pro2TraceNorm(UnfoldTT(X, dim, i)+Q{i}./beta(i), tau(i));
        Xsum = Xsum + beta(i) .*FoldTT((M{i}-Q{i}./beta(i)),dim,i);
    end
    %Update X
    Xlast = X;
    X = Xsum / betasum;
    X=mu1*(Z-N+V./p)+mu2*X;
    %Update N
    N=softthresholding(Z-X+V./p,lambda/p);
    %Update Z
    Z = X + N;
    Z(Omega)=Y(Omega);
    
    V=V+p.*(Z-X-N);
    if(p<10*beta(1))
        p=p*1.1;
    end
    for i = 1:ndims(T)
            Q{i}=Q{i}+beta(i).*(UnfoldTT(X,dim,i)-M{i});
    end
    
    errList(k) = abs(norm(X(:)-T(:))) / normT;
    if (k>50&& (abs(errList(k-1)-errList(k)) < epsilon))
        errList = errList(1:k);
        break;
    end
%     if (k>70&& (errList(k) < epsilon || errList(k)>errList(k-1)&& errList(k-1)>errList(k-2)&& errList(k-2)>errList(k-3)&&errList(k-3)>errList(k-4)))
%         errList = errList(1:k);
%         break;
%     end
    psnrList(k) = psnr(X,T);
%     loss(k) = ObjFunc(alpha,X,lambda,N);
end
fprintf('RTC-TT ends: total iterations = %d  difference=%f\n\n', k,errList(k));
end

%% Auto-weighted
function [alpha] = Weights(X,gamma)
    dim = size(X);
    for i=1:ndims(X)
        [S,V,D] = svd(UnfoldTT(X,dim,i),'econ');
        nuclear(i) = sum(V(:));
    end
    for m = 1:ndims(X)
        eta = (sum(nuclear)-2*gamma)/m;
        count = 0;
        for n = 1:ndims(X)
            if(eta<nuclear(n))
                count = count+1;
            end
        end
        if(m == count)
            break;
        end
    end
    for i=1:ndims(X)
        if(eta<nuclear(i))
        alpha(i) = (nuclear(i) - eta)/(2*gamma);
        else
        alpha(i)=0;
        end
    end
end

%% Thresholding Algorithms
function [ soft_thresh ] = softthresholding( b,lambda )
    soft_thresh = sign(b).*max(abs(b) - lambda,0);
end

%% Loss Function
function [loss] = ObjFunc(alpha,X,lambda,N)
    loss = 0;
    dim = size(X);
    for i = 1:ndims(X)
        [S,V,D] = svd(UnfoldTT(X,dim,i),'econ');
        nuclear = sum(V(:));
        loss = loss+alpha(i)*nuclear;
    end
    loss = loss + lambda*norm(N(:),1);
end