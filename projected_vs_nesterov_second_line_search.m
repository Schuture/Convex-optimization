% f(x) = g(x) + h(x) = 0.5*x'*Q*x-b'*x + IC(x)
%IC(x) is a projection
%%%%%%%%%%%%%%%%%% generate data %%%%%%%%%%%%%%%%%%
global n Q b
n = 500; 
xbar = randn(n,1); 
Q = randn(n,n);
Q = Q+Q'+eye(n); %positive definite
b = Q*xbar; 

lb = ones(n,1);
ub = 2*ones(n,1);
[x_correct,fval] = quadprog(Q,-b,[],[],[],[],lb,ub);

%%%%%%%%%%%%% projected gradient method fixed step %%%%%%
gra = zeros(100,1);
x = zeros(n,1);
beta = 0.5;
for k = 1:100
    t = 1e-2;
    grad1 = grad(x);
    y = x - t*grad1;
    while g(x) - g(y) < -t*(grad1'*grad1)
        t = beta*t;
        y = x - t*grad1;
    end
    x = proj(y);
    gra(k) = g(x);
    disp(k)
end
semilogy(gra(1:100))
hold on

%%%%%%%%% Nesterov's second method fixed step %%%%%%%%%%
gra = zeros(100,1);
x = zeros(n,1);
v = x;
beta = 0.5;
for k = 1:100
    t = 1e-2;
    theta = 2/(k+1);
    y = (1-theta)*x + theta*v;
    v_next = prox(y,v,t,theta);
    x_next = (1-theta)*x + theta*v_next;
    while g(x_next)>g(y)+grad(y)'*(x_next-y)+norm(x_next-y)^2/(2*t)
        t = beta*t;
        v_next = prox(y,v,t,theta);
        x_next = (1-theta)*x + theta*v_next;
    end
    x = x_next;
    gra(k) = g(x);
    disp(k)
end
semilogy(gra(1:100))
title('Projected gradient vs Nesterov¡¯s second method line search')
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