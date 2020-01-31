img = imread('D:\学习\课程\数值算法\2019课件\b1.JPG'); %原图
I_gray = double(rgb2gray(img)); %转换为灰度图,并转换为浮点数以便计算
[m,n]=size(I_gray); %图像的大小,m行n列

%第一步，构建权重矩阵W
sigma1 = 1;
sigma2 = 0.1;
index1 = []; %存第一个像素在W矩阵的序号
index2 = []; %存第二个像素在W矩阵的序号
values = []; %存第一个与第二个像素之间的相似度
distance = 1; %只考虑相邻的像素点
for i=1:m %图像的第i行
    for j=1:n %图像的第j列
        if i~=1 %不是第一行
            index1(end+1) = (i-1)*n+j;
            index2(end+1) = (i-2)*n+j; %相同列上面那一行像素
            value = exp(-(I_gray(i,j)-I_gray(i-1,j))^2/sigma1-1/sigma2);
            values(end+1) = value;
        end
        if i~=m %不是最后一行
            index1(end+1) = (i-1)*n+j;
            index2(end+1) = i*n+j; %相同列下面那一行像素
            value = exp(-(I_gray(i,j)-I_gray(i+1,j))^2/sigma1-1/sigma2);
            values(end+1) = value;
        end
        if j~=1 %不是第一列
            index1(end+1) = (i-1)*n+j;
            index2(end+1) = (i-1)*n+j-1; %相同行左边那一列像素
            value = exp(-(I_gray(i,j)-I_gray(i,j-1))^2/sigma1-1/sigma2);
            values(end+1) = value;
        end
        if j~=n %不是最后一列
            index1(end+1) = (i-1)*n+j;
            index2(end+1) = (i-1)*n+j+1; %相同行右边那一列像素
            value = exp(-(I_gray(i,j)-I_gray(i,j+1))^2/sigma1-1/sigma2);
            values(end+1) = value;
        end
    end
end
W = sparse(index1',index2',values');

%第二步，解特征值问题,待求的是D^-0.5*（D - W）*D^-0.5的最小特征值和对应的特征向量
d = sum(W,2); %W按行求和
D = sparse([1:m*n]',[1:m*n]',d); %由d生成的对角稀疏矩阵
L = D - W;
D_prime = sparse([1:m*n]',[1:m*n]',1./sqrt(d)); %D^(-0.5)
A = D_prime*L*D_prime;
[x,y] = eigs(A,1,1e-14); %求矩阵的最小特征值和对应特征向量，x为特征向量矩阵，y为特征值矩阵。

%第三步，得到矩阵最小特征值对应的特征向量
x_lambda = x;

%第四步，将特征向量重新赋值并且重整回图像大小
for i=1:m*n
    if x_lambda(i)>0
        x_lambda(i) = 1;
    else
        x_lambda(i) = 0;
    end
end
img_new = reshape(x_lambda,[n,m]);
subplot(121),imshow(img);
subplot(122),imshow(img_new');