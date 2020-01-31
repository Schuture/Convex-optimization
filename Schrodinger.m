% ��Ѧ���̷��� -hbar^2/2m*psi''(x)+V(x)psi(x)=E*psi(x)
%%%%%%%%%%%%%%%%%%%% ���峣�� %%%%%%%%%%%%%%%%%%%%%%
me = 9.10938188e-31; % mass of electron
eV = 1.60217646e-19; % electricity capacity
h = 6.626068e-34; % Plank constant
hbar = 1.05457148e-34; % hbar=h/2/pi 
 
%%%%%%%%%%% �����������Լ���ȡ���ַ����ֳ�128��С��
a = 10e-9;                   % ȫ�ռ���
n = 128;                     % �������
z = linspace(-a/2,a/2,n);    % ���λ�õ�����
dz = a/n;                    % һ�������

K = 1; % K�ڵ�����ѧ���ǵ���ϵ������������m*omega^2
V = 1/2*K*z'.^2; %���ܺ���V(z)=(K*z^2)/2
%spdiags(B,d,m,n) ͨ����ȡ B ���в��� d ָ���ĶԽ��߷������ǣ�������һ�� m��n ϡ�����
pmatrix = spdiags(V,0,n,n);  % nxn�Խ����ܾ���

% Schrodinger matrix used to get the wavefuction 
vector = zeros(n,3); 
vector(1:n,1) = -hbar^2/(2*me)/dz^2; 
vector(1:n,2) = 2*hbar^2/(2*me)/dz^2; 
vector(2:n,3) = -hbar^2/(2*me)/dz^2; 
vmatrix = spdiags(vector,-1:1,n,n); % nxn���ԽǾ���
matrix = pmatrix+vmatrix;

eignum = 10;     % Ҫ���������ֵ����

% �������������������̣�ȡ��С�ļ�������ֵ
[eigvector, eigvalue] = eigs(matrix,eignum,0); 
diag(eigvalue)/eV         % ����
 
 for i = 1:eignum  % ��eignum����������ͼ��
     wavefunction = eigvector(:,i);  %̬�ܶȲ�������ĳ��������δ��һ��
     energy = eigvalue(i,i); 
      
     % ��һ��������
     wavefunction = wavefunction/sqrt(sum(abs(wavefunction.^2)*dz)); %̬�ܶ�ƽ�����ֲ���1
      
     % ��ͼ
     figure(1); 
     subplot(eignum/2,2,i),plot(z,wavefunction); 
     figure(2); 
     plot(i,energy,'*'); 
     hold on
 end 