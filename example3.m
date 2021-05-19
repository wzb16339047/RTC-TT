addpath(genpath(cd))
clear
%% Examples for testing the robust low-rank tensor completion

% Initial parameters
Nway = [4 4 4 4 4 4 4 4 3];     % 9th-order dimensions for KA
n1 = 256;                       % dimensions
n2 = 256;
n3 = 3;
I1 = 2; J1 = 2;                 % KA parameters
maxIter=1000;
epsilon = 1e-5;

mr = 0.5;                       % Missing ratio
noise = 0.1;                    % Noisy ratio

T = double(imread('peppers.bmp'));                                           % ÑÕÉ«£¬[0,0,0]ºÚÉ«£»[255,255,255]°×É«£»
loc = int32([0 0;0 190;0 100]); 
Y = insertText(T,loc,"1234567",'FontSize',50, 'BoxOpacity',0);
T=T./255;
Y=Y./255;
% XX = step( textInserter, A, TemText, loc );
Y = imnoise(Y,'salt & pepper',noise);

% Y=imnoise(imread('peppers.bmp'),'salt & pepper',noise);
% Y = double(Y);

figure(1);
imshow(T);
figure(2);
imshow(Y)


%Ket Augmentation
T = CastImageAsKet(T,Nway,I1,J1);
Y = CastImageAsKet(Y,Nway,I1,J1);
% Known
% P = round((1-mr)*prod(Nway));
% Known = randsample(prod(Nway),P);
% [Known,~] = sort(Known);
Known=logical(T==Y);
mr = 1-sum(Known(:))/prod(Nway)


gg = 1.1
ttpsnr = ones(1,8);

%% RTC
for i = 1:1
    [X_rtc, N_rtc, Show, alpha,errList,psnrList] = RTC_alpha(T,Y, Known, maxIter, epsilon, mr, noise,gg(i));
    X_rtc=CastKet2Image(X_rtc,n1,n2,I1,J1);
    T = CastKet2Image(T,n1,n2,I1,J1);
    N_rtc=CastKet2Image(N_rtc,n1,n2,I1,J1);
    Show=CastKet2Image(Show,n1,n2,I1,J1);
    ttpsnr(i) = psnr(X_rtc,T);
%     T = CastImageAsKet(T,Nway,I1,J1);
end
% Show Pictures
figure(1);
imshow(Show);
title('Observation');
figure(2);
imshow(X_rtc);
title('X');
psnr(X_rtc,T)
disp(alpha)
psnr(X_rtc,T)