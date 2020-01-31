% 逻辑回归，不存在h(x)不光滑函数
%%%%%%%%%%%%%%%%%%generate the data%%%%%%%%%%%%%%%%%%
global m n A b 
m = 500; 
n = 1000; 
A = randn(n,m);  % a is the column of matrix A
b = sign(rand(m,1)-0.5); 

%%%%%%%%%%%%%%%%%%gradient descent%%%%%%%%%%%%%%%%%%
x = zeros(n+1,1); % omega is the first n elements
alpha = 3e-3;
norm_grad = zeros(10000,1);
for k = 1:10000
    grad1 = grad(x);
    norm_grad(k) = norm(grad1);
    x = x-alpha*grad1;
    disp(k)
    if norm_grad(k)<1e-4
        break
    end
end
semilogy(norm_grad(1:10000))
hold on

%%%%%%%%%%%%%%%%%%FISTA%%%%%%%%%%%%%%%%%%
t = alpha;
norm_grad = zeros(10000,1);
x = zeros(n+1,1); % omega is the first n elements
x_last = x;
x_next = zeros(n+1,1);
for k = 1:10000
    y = x+(k-2)/(k+1)*(x-x_last);
    x_next = prox(y,t);
    norm_grad(k) = norm(grad(y));
    disp(k)
    if norm_grad(k)<1e-4
        break
    end
    x_last = x;
    x = x_next;
end
semilogy(norm_grad(1:10000))
hold on
title('GD vs FISTA')
xlabel('iteration');
ylabel('norm of gradient')
legend('gradient descent','FISTA')

%%%%%%%%%%%%%%%%%%define functions%%%%%%%%%%%%%%%%%%
function [ret] = f(x)
global m A b
ret = 0;
for i=1:m
    ret = ret + log(1+exp(-b(i)*x(1:n)'*A(:,i)+x(n+1)));
end
ret = ret/m;
end

function [ret] = grad(x)
global m n A b
ret = zeros(n+1,1);
for i=1:m
    ret(1:n) = ret(1:n) + ...
        -b(i)*A(:,i)*exp(-b(i)*x(1:n)'*A(:,i)+x(n+1))/(1+exp(-b(i)*x(1:n)'*A(:,i)+x(n+1)));
end
for i=1:m
    ret(n+1) = ret(n+1) + ...
        exp(-b(i)*x(1:n)'*A(:,i)+x(n+1))/(1+exp(-b(i)*x(1:n)'*A(:,i)+x(n+1)));
end
ret = ret/m;
end

function [ret] = prox(y,t)
ret = y - t*grad(y);
end