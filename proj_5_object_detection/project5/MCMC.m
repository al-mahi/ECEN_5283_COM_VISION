clear all;
I=double(imread('discs1.bmp'))/255;     % read the test image
T=double(imread('target.bmp'))/255;     % read the target image
[X Y]=size(I); Z=zeros(X,Y);
Kmax=1000;                               % the number of random walks

Oxy=round(rand(1,2)*X)+1;               % an initial position
L1=likelihood(I,T,Oxy,1);               % the initial likelihood
Io=drawcircle(I,Oxy,1);                 % locate the object
figure(1),imshow(Io);
Imframe(1:X,1:Y,1)=Io; Imframe(1:X,1:Y,2)=Io; Imframe(1:X,1:Y,3)=Io;
videoseg(1)=im2frame(Imframe);          % make the first frame

for i=1:Kmax
    Dxy=Oxy+round(randn(1,2)*20);       % random walk
    Dxy=clip(Dxy,1,X);                  % make sure the position in the image
    L2=likelihood(I,T,Dxy,1);           % evaluate the likelihood
    v=min(1,L2/L1);                     % compute the acceptance ratio
    u=rand;                             % draw a sample uniformly in [0 1]
    if v>u
        Oxy=Dxy;     L1=L2;             % accept the move
        Io=drawcircle(I,Oxy,1);         % draw the new position
    end
    figure(1),imshow(Io);
    Imframe(1:X,1:Y,1)=Io; Imframe(1:X,1:Y,2)=Io; Imframe(1:X,1:Y,3)=Io;
    videoseg(i+1)=im2frame(Imframe); 
end
movie2avi(videoseg(1:(Kmax+1)),'MCMC1.avi','FPS',10,'COMPRESSION','None');

