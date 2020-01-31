%最小二乘问题 min|Ax-b| 解为inv(A'A)A' b

%设置A*x_correct=b，和迭代初值x0
A = randn(1000);
B = A'*A; %对称正定
x_correct = ones(1000,1);
b = A*x_correct;
x0 = rand(1000,1);

Q = []; %正交矩阵,1000*k
r = []; %残差,每一列代表一次迭代的残差
H = []; %上Hessenburg阵，(k+1)*k
x = []; %每一步计算出来的x,1000*k
y = []; %每一步的更新y
r0 = b-A*x0; %初始残差，1000*1
H10 = norm(r0);
resn = []; %残差的范数

%把k=0的情况单独写在前面
Q(:,1) = r0/H10;
r(:,1) = A*Q(:,1);
H(1,1) = Q(:,1)'*r(:,1);
r(:,1) = r(:,1)-H(1,1)*Q(:,1);
H(2,1) = norm(r(:,1));
e1 = zeros(2,1);
e1(1) = 1;
y(:,1) = inv(H'*H)*H'*e1*H10; %y使得|H10e1-Hy|取最小值
x(:,1) = x0+Q*y(:,1);
resn(1) = norm(b-A*x(:,1));

k = 1;
while H(k+1,k)>1
    Q(:,k+1) = r(:,k)/H(k+1,k);
    k = k+1;
    r(:,k) = A*Q(:,k);
    for i=1:k
        H(i,k) = Q(:,i)'*r(:,k);
        r(:,k) = r(:,k)-H(i,k)*Q(:,i);
    end
    H(k+1,k) = norm(r(:,k));
    e1 = zeros(k+1,1);
    e1(1) = 1;
    y(1:k,k) = inv(H'*H)*H'*e1*H10; %y使得|H10e1-Hy|取最小值
    x(:,k) = x0+Q*y(:,k);
    resn(k) = norm(b-A*x(:,k));
    
    if k>1000 %最多迭代1000次
        break
    end
end
result = x(:,k);
disp('iteration:')
disp(k)
disp('final residual:')
disp(norm(b-A*result))
semilogy(1:k,resn)
xlabel('iteration')
ylabel('norm of residual')