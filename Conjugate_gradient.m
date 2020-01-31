%%%%%%%%%%%%%%%%%%generate data%%%%%%%%%%%%%%%%%%%
n = 100; 
L = sparse(eye(n) - diag(ones(n-1, 1), 1)); 
A = kron(L + L', eye(n)) + kron(eye(n), L + L'); %10000*10000
b = ones(n*n, 1); 

%%%%%%%%%%%%%%%%%%iteration%%%%%%%%%%%%%%%%%%%
x = zeros(n*n,1);
r = zeros(n*n,10000);
res = zeros(10001);

r0 = b-A*x;
res(1) = norm(r0);
p = r0;

alpha = r0'*r0/(p'*A*p);
x = x + p*alpha;
r(:,1) = r0 - A*p*alpha;
res(2) = norm(r(:,1));
beta = r(:,1)'*r(:,1)/(r0'*r0);
p = r(:,1) + p*beta;

for k = 1:9999
    Ap = A*p; %save computation
    alpha = r(:,k)'*r(:,k)/(p'*Ap);
    x = x + p*alpha;
    r(:,k+1) = r(:,k) - Ap*alpha;
    res(k+2) = norm(r(:,k+1));
    beta = r(:,k+1)'*r(:,k+1)/(r(:,k)'*r(:,k));
    p = r(:,k+1) + p*beta;
    disp(k)
end
semilogy(res(1:10001))
title('Conjugate gradient method')
xlabel('Iteration')
ylabel('Residual')