%%%%%%%%%%%%%initialize data%%%%%%%%%%%%%
global A b tau
m = 100; 
n = 500; 
s = 50; 
A = randn(m,n); 
xs = zeros(n,1);  %500*1
picks = randperm(n); 
xs(picks(1:s)) = randn(s,1);
b = A*xs;
tau=0.001;

%%%%%%%%%%%%%cvx solution%%%%%%%%%%%%%
cvx_begin %find the minimum of f
variable x_correct(500)
minimize(0.5*square_pos(norm(A*x_correct-b))+tau*norm(x_correct,1))
cvx_end
f_optimal = cvx_optval;
disp('f_optimal:')
disp(f_optimal)

%%%%%%%%%%%%%Polyak¡¯s step size%%%%%%%%%%%%%
x = zeros(500,1);
diff = zeros(10000,1);
for k = 1:10000
    grad1 = grad(x);
    diff(k) = f(x)-f_optimal;
x = x-diff(k)*grad1/norm(grad1)^2;
end
semilogy(diff(1:10000))
title('Subgradient method with Polyak¡¯s step size ')
xlabel('Iteration')
ylabel('Difference between f(x) and f*')


function [ret]=f(x)
global A b tau
ret = 0.5*norm(A*x-b)^2+tau*norm(x,1);
end

function [ret]=grad(x)
global A b tau
gradnorm = zeros(500,1);
for i=1:500
    if x(i)>0
        gradnorm(i) = 1;
    end
    if x(i)<0
        gradnorm(i) = -1;
    end
end
ret = A'*(A*x-b)+tau*gradnorm;
end