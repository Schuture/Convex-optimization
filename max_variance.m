img = imread('D:\ѧϰ\�γ�\��ֵ�㷨\2019�μ�\b3.JPG');%ԭͼ
I_gray=rgb2gray(img);%ת��Ϊ�Ҷ�ͼ

subplot(121),imshow(img);
%ת��Ϊ˫����
I_double=double(I_gray);
[wid,len]=size(I_gray);%ͼ��Ĵ�С
%�Ҷȼ�
colorLevel=256;
%ֱ��ͼ
hist=zeros(colorLevel,1);
%����ֱ��ͼ
for i=1:wid
    for j=1:len
        m=I_gray(i,j)+1;%ͼ��ĻҶȼ�m
        hist(m)=hist(m)+1;%�Ҷ�ֵΪi�����غ�
    end
end
%ֱ��ͼ��һ��
hist=hist/(wid*len);%���Ҷ�ֵ���� Pi
miuT=0;%���������ֵ
for m=1:colorLevel
    miuT=miuT+(m-1)*hist(m);  %�����ֵ
end
xigmaB2=0;%
for mindex=1:colorLevel
    threshold=mindex-1;%�趨��ֵ
    omega1=0;%Ŀ�����
    omega2=0;%��������
    for m=1:threshold-1
        omega1=omega1+hist(m);% Ŀ����� W0
    end
    omega2=1-omega1; %�����ĸ��� W1
    miu1=0;%Ŀ���ƽ���Ҷ�ֵ
    miu2=0;%������ƽ���Ҷ�ֵ
    for m=1:colorLevel
        if m<threshold
            miu1=miu1+(m-1)*hist(m);%Ŀ�� i*pi���ۼ�ֵ[1 threshold]
        else
            miu2=miu2+(m-1)*hist(m);%���� i*pi���ۼ�ֵ[threshold m]
        end
    end
    miu1=miu1/omega1;%Ŀ���ƽ���Ҷ�ֵ
    miu2=miu2/omega2;%������ƽ���Ҷ�ֵ
    xigmaB21=omega1*(miu1-miuT)^2+omega2*(miu2-miuT)^2;%��󷽲�
    xigma(mindex)=xigmaB21;%���趨һ��ֵ �ٱ������лҶȼ�
    %�ҵ�xigmaB21��ֵ���
    if xigmaB21>xigmaB2
        finalT=threshold;%�ҵ���ֵ �Ҷȼ�
        xigmaB2=xigmaB21;%����Ϊ���
    end
end
%��ֵ��һ��
fT=finalT/255;
for i=1:wid
     for j=1:len
         if I_double(i,j)>finalT %�������趨�ľ�ֵ ��ΪĿ��
             bin(i,j)=0;
         else
             bin(i,j)=1;
         end
     end
end
subplot(122),imshow(bin);