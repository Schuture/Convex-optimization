% 解薛定谔方程 -hbar^2/2m*psi''(x)+V(x)psi(x)=E*psi(x)
%%%%%%%%%%%%%%%%%%%% 定义常数 %%%%%%%%%%%%%%%%%%%%%%
me = 9.10938188e-31; % mass of electron
eV = 1.60217646e-19; % electricity capacity
h = 6.626068e-34; % Plank constant
hbar = 1.05457148e-34; % hbar=h/2/pi 
 
%%%%%%%%%%% 定义格点数量以及宽度。差分法，分成128个小份
a = 10e-9;                   % 全空间宽度
n = 128;                     % 格点数量
z = linspace(-a/2,a/2,n);    % 格点位置的向量
dz = a/n;                    % 一个格点宽度

K = 1; % K在弹性力学中是弹性系数，在这里是m*omega^2
V = 1/2*K*z'.^2; %势能函数V(z)=(K*z^2)/2
%spdiags(B,d,m,n) 通过获取 B 的列并沿 d 指定的对角线放置它们，来创建一个 m×n 稀疏矩阵。
pmatrix = spdiags(V,0,n,n);  % nxn对角势能矩阵

% Schrodinger matrix used to get the wavefuction 
vector = zeros(n,3); 
vector(1:n,1) = -hbar^2/(2*me)/dz^2; 
vector(1:n,2) = 2*hbar^2/(2*me)/dz^2; 
vector(2:n,3) = -hbar^2/(2*me)/dz^2; 
vmatrix = spdiags(vector,-1:1,n,n); % nxn三对角矩阵
matrix = pmatrix+vmatrix;

eignum = 10;     % 要计算的特征值数量

% 用特征函数解特征方程，取最小的几个特征值
[eigvector, eigvalue] = eigs(matrix,eignum,0); 
diag(eigvalue)/eV         % 测试
 
 for i = 1:eignum  % 画eignum幅波函数的图像
     wavefunction = eigvector(:,i);  %态密度波函数的某个倍数，未归一化
     energy = eigvalue(i,i); 
      
     % 归一化波函数
     wavefunction = wavefunction/sqrt(sum(abs(wavefunction.^2)*dz)); %态密度平方积分才是1
      
     % 画图
     figure(1); 
     subplot(eignum/2,2,i),plot(z,wavefunction); 
     figure(2); 
     plot(i,energy,'*'); 
     hold on
 end 