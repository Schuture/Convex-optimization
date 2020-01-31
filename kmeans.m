img = imread('D:\学习\课程\数值算法\2019课件\b2.JPG');%原图
I_gray=rgb2gray(img);%转换为灰度图
[mu,mask] = Kmeans(I_gray,2);
subplot(121),imshow(img);
subplot(122),imshow(mask);

function [mu,mask]=Kmeans(ima,k)
%功能·：运用K-means算法对图像进行分割
% 输入 ima-输入的灰度图像 K-分类数
%输出 mu -均值类向量  mask-分类后的图像
ima=double(ima);
copy=ima;
ima=ima(:);
mi=min(ima);%找到最小值
ima=ima-mi+1;
s=length(ima);%有多少灰度级
%计算图像灰度直方图
m=max(ima)+1;%图像最大灰度值
h=zeros(1,m);
hc=zeros(1,m);
for i=1:s
    if (ima(i)>0)
        h(ima(i))=h(ima(i))+1;%灰度值i累加
    end
end
ind =find(h);
h1=length(ind);
%初始化质心
mu=(1:k)*m/(k+1);
%start process
while(true)
    oldmu=mu;
    %现有的分类
    for i=1:h1
        c=abs(ind(i)-mu);
        cc=find(c==min(c));
        hc(ind(i))=cc(1);
    end
    %重新计算均值
    for i=1:k
        a=find(hc==i);
        mu(i)=sum(a.*h(a))/sum(h(a));
    end
    if(mu==oldmu) 
        break;
    end
end
    %计算生成分类后的图像
s=size(copy);
mask=zeros(s);
for i=1:s(1)
    for j=1:s(2)
        c=abs(copy(i,j)-mu);
        a=find(c==min(c));
        mask(i,j)=a(1);
    end
end
end