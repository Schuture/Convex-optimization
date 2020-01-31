img = imread('D:\ѧϰ\�γ�\��ֵ�㷨\2019�μ�\b1.JPG'); %ԭͼ
I_gray = double(rgb2gray(img)); %ת��Ϊ�Ҷ�ͼ,��ת��Ϊ�������Ա����
[m,n]=size(I_gray); %ͼ��Ĵ�С,m��n��

%��һ��������Ȩ�ؾ���W
sigma1 = 1;
sigma2 = 0.1;
index1 = []; %���һ��������W��������
index2 = []; %��ڶ���������W��������
values = []; %���һ����ڶ�������֮������ƶ�
distance = 1; %ֻ�������ڵ����ص�
for i=1:m %ͼ��ĵ�i��
    for j=1:n %ͼ��ĵ�j��
        if i~=1 %���ǵ�һ��
            index1(end+1) = (i-1)*n+j;
            index2(end+1) = (i-2)*n+j; %��ͬ��������һ������
            value = exp(-(I_gray(i,j)-I_gray(i-1,j))^2/sigma1-1/sigma2);
            values(end+1) = value;
        end
        if i~=m %�������һ��
            index1(end+1) = (i-1)*n+j;
            index2(end+1) = i*n+j; %��ͬ��������һ������
            value = exp(-(I_gray(i,j)-I_gray(i+1,j))^2/sigma1-1/sigma2);
            values(end+1) = value;
        end
        if j~=1 %���ǵ�һ��
            index1(end+1) = (i-1)*n+j;
            index2(end+1) = (i-1)*n+j-1; %��ͬ�������һ������
            value = exp(-(I_gray(i,j)-I_gray(i,j-1))^2/sigma1-1/sigma2);
            values(end+1) = value;
        end
        if j~=n %�������һ��
            index1(end+1) = (i-1)*n+j;
            index2(end+1) = (i-1)*n+j+1; %��ͬ���ұ���һ������
            value = exp(-(I_gray(i,j)-I_gray(i,j+1))^2/sigma1-1/sigma2);
            values(end+1) = value;
        end
    end
end
W = sparse(index1',index2',values');

%�ڶ�����������ֵ����,�������D^-0.5*��D - W��*D^-0.5����С����ֵ�Ͷ�Ӧ����������
d = sum(W,2); %W�������
D = sparse([1:m*n]',[1:m*n]',d); %��d���ɵĶԽ�ϡ�����
L = D - W;
D_prime = sparse([1:m*n]',[1:m*n]',1./sqrt(d)); %D^(-0.5)
A = D_prime*L*D_prime;
[x,y] = eigs(A,1,1e-14); %��������С����ֵ�Ͷ�Ӧ����������xΪ������������yΪ����ֵ����

%���������õ�������С����ֵ��Ӧ����������
x_lambda = x;

%���Ĳ����������������¸�ֵ����������ͼ���С
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