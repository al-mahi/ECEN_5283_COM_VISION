clear all;                          % using direct (important sampling) 
I=double(imread('discs1.bmp'))/255; % read the test image
T=double(imread('target.bmp'))/255; % read the target image
[X Y]=size(I);
Z=zeros(X,Y);

Kmax=2000;                          % the number of samples    
Posi=[0 0];                         % position variable

for i=1:Kmax
    Axy=round(rand(1,2)*X)+1;       % draw a 2-D random sample uniformly
    Li(i)=likelihood(I,T,Axy,1);    % evaluate the likelihood
    Px(i,:)=Axy(1:2);               % save the 2-D position hypothesis
    Z(Axy(1),Axy(2))=Li(i);         % save the likelihood for that 2-D position hypothesis
    Posi=Posi+Li(i)*Px(i,:);        % compute the weighted mean estimation
end

Posi=round(Posi/sum(Li));           % compute the weighted mean estimation
Z=Z/sum(Li);                        % compute the normalized distribution

J=drawcircle(I,Posi,1);             % locate the object according to the mean estimation

figure(1), mesh(Z);                 % draw the estimated distribution
figure(2), imshow(J);               % show the object detection result

