%%%%%%%%%%%%%%%%%%generate data%%%%%%%%%%%%%%%%%%
global m n s A b tau
m = 1000; 
n = 500; 
s = 50; 
A = randn(m,n); 
xs = zeros(n,1); 
picks = randperm(n); 
xs(picks(1:s)) = randn(s,1); 
b = A*xs;
tau = 1e-6;
 
%%%%%%%%%%%%%%%%%%proximal fixed step size%%%%%%%%%%%%%%%%%%
L = norm(A'*A);
t = 1/L;
diff = zeros(200,1);
x = zeros(n,1);
x_last = x;
x_next = zeros(n,1);
for k = 1:200
    x_next = prox(x,t);
    diff(k) = norm(x-xs);
    disp(k)
    x_last = x;
    x = x_next;
end
semilogy(diff(1:200))
hold on
 
%%%%%%%%%%%%%%%%%%FISTA fixed step size%%%%%%%%%%%%%%%%%%
diff = zeros(200,1);
x = zeros(n,1);
x_last = x;
x_next = zeros(n,1);
for k = 1:200
    y = x+(k-2)/(k+1)*(x-x_last);
    x_next = prox(y,t);
    diff(k) = norm(x-xs);
    disp(k)
    x_last = x;
    x = x_next;
end
semilogy(diff(1:200))
title('Proximal gradient vs FISTA with step size 1/L')
legend('Proximal gradient','FISTA')
xlabel('Iteration')
ylabel('Difference between x and xs')

%%%%%%%%%%%%%%define functions%%%%%%%%%%%%%%
function [ret]=f(x)
global A b tau
ret = 0.5*norm(A*x-b)^2+tau*norm(x,1);
end

function [ret]=grad(x)
global A b
ret = A'*(A*x-b);
end

function [ret] = prox(y,t)
global n tau
z = y-t*grad(y);
ret = zeros(n,1);
for i=1:n
    if z(i)>t*tau
        ret(i) = z(i)-t*tau;
    end
    if z(i)<-t*tau
        ret(i) = 0;
    end
    if -t*tau<z(i)<t*tau
        ret(i) = z(i)+t*tau;
    end
end
end