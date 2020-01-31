% f(x)=g(x)+h(x)=norm(Ax-b)+tau*|x|
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
 
%%%%%%%%%%%%%%%%%%proximal line search%%%%%%%%%%%%%%%%%%
beta = 0.5;
diff = zeros(200,1);
x = zeros(n,1);
for k = 1:200
    t = 1e-2;
    Gt = (x-prox(x,t))/t;
    while g(x-t*Gt)>g(x)-t*grad(x)'*Gt+0.5*t*norm(Gt)^2
        t = t*beta;
        Gt = (x-prox(x,t))/t;
    end
    x = prox(x,t);
    diff(k) = norm(x-xs);
    disp(k)
end
semilogy(diff(1:200))
hold on

%%%%%%%%%%%%%%%%%%FISTA line search%%%%%%%%%%%%%%%%%%
diff = zeros(200,1);
x = zeros(n,1);
x_last = x;
t = 1;
for k = 1:200
    y = x+(k-2)/(k+1)*(x-x_last);
    x_last = x;
    x = prox(y,t);
    while g(x)>g(y)+grad(y)'*(x-y)+norm(x-y)^2/(2*t)
        t = t*beta;
        x = prox(y,t);
    end
    diff(k) = norm(x-xs);
    disp(k)
end
semilogy(diff(1:200))
title('Proximal gradient vs FISTA with line search')
legend('Proximal gradient','FISTA')
xlabel('Iteration')
ylabel('Difference between x and xs')

%%%%%%%%%%%%%%define functions%%%%%%%%%%%%%%
function [ret]=g(x)
global A b
ret = 0.5*norm(A*x-b)^2;
end
 
function [ret]=grad(x) % gµÄÌİ¶È
global A b
ret = A'*(A*x-b);
end
 
function [ret] = prox(y,t) %prox(y-t*grad(y))
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