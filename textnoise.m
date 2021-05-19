X0 = imread('lena.bmp');
A  = uint8(ones(size(X0))*255);
TemText = uint8(['345678' 0 '456789']);% 'abc' 或 '345'等是要产生的数字或字母
textColor = [ 0, 0, 0 ];                                            % 颜色，[0,0,0]黑色；[255,255,255]白色；
loc = int32([5 50;]); 
textInserter = insertText(X0,loc,"ABCDEFG",'FontSize',50, 'BoxOpacity',0);
% XX = step( textInserter, A, TemText, loc );
imshow(textInserter)