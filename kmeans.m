img = imread('D:\ѧϰ\�γ�\��ֵ�㷨\2019�μ�\b2.JPG');%ԭͼ
I_gray=rgb2gray(img);%ת��Ϊ�Ҷ�ͼ
[mu,mask] = Kmeans(I_gray,2);
subplot(121),imshow(img);
subplot(122),imshow(mask);

function [mu,mask]=Kmeans(ima,k)
%���ܡ�������K-means�㷨��ͼ����зָ�
% ���� ima-����ĻҶ�ͼ�� K-������
%��� mu -��ֵ������  mask-������ͼ��
ima=double(ima);
copy=ima;
ima=ima(:);
mi=min(ima);%�ҵ���Сֵ
ima=ima-mi+1;
s=length(ima);%�ж��ٻҶȼ�
%����ͼ��Ҷ�ֱ��ͼ
m=max(ima)+1;%ͼ�����Ҷ�ֵ
h=zeros(1,m);
hc=zeros(1,m);
for i=1:s
    if (ima(i)>0)
        h(ima(i))=h(ima(i))+1;%�Ҷ�ֵi�ۼ�
    end
end
ind =find(h);
h1=length(ind);
%��ʼ������
mu=(1:k)*m/(k+1);
%start process
while(true)
    oldmu=mu;
    %���еķ���
    for i=1:h1
        c=abs(ind(i)-mu);
        cc=find(c==min(c));
        hc(ind(i))=cc(1);
    end
    %���¼����ֵ
    for i=1:k
        a=find(hc==i);
        mu(i)=sum(a.*h(a))/sum(h(a));
    end
    if(mu==oldmu) 
        break;
    end
end
    %�������ɷ�����ͼ��
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