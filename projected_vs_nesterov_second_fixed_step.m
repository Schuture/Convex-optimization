% f(x) = g(x) + h(x) = 0.5*x'*Q*x-b'*x + IC(x)
%IC(x) is a projection
%%%%%%%%%%%%%%%%%% generate data %%%%%%%%%%%%%%%%%%
global n Q b
n = 500; 
xbar = randn(n,1); 
Q = randn(n,n);
Q = Q+Q'+eye(n); %positive definite
b = Q*xbar; 
 
L = norm(Q');
t = 1/L;
%%%%%%%%%%%%% projected gradient method fixed step %%%%%%
gra = zeros(50,1);
x = zeros(n,1);
for k = 1:50
    grad1 = grad(x);
    y = x - t*grad1;
    x = proj(y);
    gra(k) = g(x);
    disp(k)
end
semilogy(gra(1:50))
hold on
 
%%%%%%%%% Nesterov¡¯s second method fixed step %%%%%%%%%%
gra = zeros(50,1);
x = zeros(n,1);
v = x;
for k = 1:50
    theta = 2/(k+1);
    y = (1-theta)*x + theta*v;
    v_next = prox(y,v,t,theta);
    x = (1-theta)*x + theta*v_next;
    gra(k) = g(x);
    disp(k)
end
semilogy(gra(1:50))
title('Projected gradient vs Nesterov¡¯s second method with 1/L step')
legend('Projected gradient','Nesterov¡¯s second method')
xlabel('Iteration')
ylabel('function value')
 
%%%%%%%%%%%%%%define functions%%%%%%%%%%%%%%
function [ret]=g(x)
global Q b
ret = 0.5*x'*Q*x-b'*x;
end
 
function [ret]=grad(x) % gradient of g
global Q b
ret = Q*x-b;
end
 
function [ret] = prox(y,v,t,theta) %prox(v-t*grad(y)/theta)
global n 
ret = v-t*grad(y)/theta;
for i=1:n % do projection
    if ret(i)>2
        ret(i) = 2;
    end
    if ret(i)<1
        ret(i) = 1;
    end
end
end
 
function [ret] = proj(y) %proj(y)
global n 
ret = y;
for i=1:n % do projection
    if ret(i)>2
        ret(i) = 2;
    end
    if ret(i)<1
        ret(i) = 1;
    end
end
end
