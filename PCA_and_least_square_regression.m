%%%%%%%%%%%%%% read data %%%%%%%%%%%%%%%%
filename = 'data.txt';
[x,y]=textread(filename,'%n%n',100);
plot(x,y,'o')
hold on

%%%%%%%%%%%%%% PCA %%%%%%%%%%%%%%%%
data = [x';y']; % data should be columnwise
b = [mean(x);mean(y)];
data = data - b; % we do SVD with the 2*100 matrix 
[U,S,V] = svd(data,'econ');
direction = U(:,1);
k = direction(2)/direction(1); % slope
c = b(2) - k*b(1); %intercept
f = @(x)(k*x+c);
line([0,4],[f(0),f(4)],'linestyle','--','color','k')
hold on

%%%%%%%%%%%%%% least square %%%%%%%%%%%%%%%%
k = x'*y/(x'*x); % slope
c = b(2) - k*b(1); %intercept
f = @(x)(k*x+c);
line([0,4],[f(0),f(4)],'linestyle','--','color','r')
hold on
legend('data','PCA','Least square')
title('Linear regression')
xlabel('x')
ylabel('y')