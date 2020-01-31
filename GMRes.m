%��С�������� min|Ax-b| ��Ϊinv(A'A)A' b

%����A*x_correct=b���͵�����ֵx0
A = randn(1000);
B = A'*A; %�Գ�����
x_correct = ones(1000,1);
b = A*x_correct;
x0 = rand(1000,1);

Q = []; %��������,1000*k
r = []; %�в�,ÿһ�д���һ�ε����Ĳв�
H = []; %��Hessenburg��(k+1)*k
x = []; %ÿһ�����������x,1000*k
y = []; %ÿһ���ĸ���y
r0 = b-A*x0; %��ʼ�в1000*1
H10 = norm(r0);
resn = []; %�в�ķ���

%��k=0���������д��ǰ��
Q(:,1) = r0/H10;
r(:,1) = A*Q(:,1);
H(1,1) = Q(:,1)'*r(:,1);
r(:,1) = r(:,1)-H(1,1)*Q(:,1);
H(2,1) = norm(r(:,1));
e1 = zeros(2,1);
e1(1) = 1;
y(:,1) = inv(H'*H)*H'*e1*H10; %yʹ��|H10e1-Hy|ȡ��Сֵ
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
    y(1:k,k) = inv(H'*H)*H'*e1*H10; %yʹ��|H10e1-Hy|ȡ��Сֵ
    x(:,k) = x0+Q*y(:,k);
    resn(k) = norm(b-A*x(:,k));
    
    if k>1000 %������1000��
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