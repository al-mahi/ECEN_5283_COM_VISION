T=imread('test_image.bmp');

for i=1:100
L=T;
% some processing in L
Frame(:,:,1)=L/i*2;  % Red channel
Frame(:,:,2)=L/i*2;  % Blue channel
Frame(:,:,3)=L/i*2;  % Green channel
Mo(i)=im2frame(Frame)
end
Movie2avi(Mo,'filename.avi');

