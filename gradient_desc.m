f=@(x)( exp(x(1)+3*x(2)-0.1)+exp(x(1)-3*x(2)-0.1)+exp(-x(1)-0.1)+0.1*x'* x);
grad=@(x)([exp(x(1)+3*x(2)-0.1)+exp(x(1)-3*x(2)-0.1)-exp(-x(1)-0.1)+0.2*x(1);
           3*exp(x(1)+3*x(2)-0.1)-3*exp(x(1)-3*x(2)-0.1)+0.2*x(2)]);
         
hessian=@(x)( [exp(x(1)+3*x(2)-0.1)+exp(x(1)-3*x(2)-0.1)+exp(-x(1)-0.1)+0.2, ...
        3*exp(x(1)+3*x(2)-0.1)-3*exp(x(1)-3*x(2)-0.1);
        3*exp(x(1)+3*x(2)-0.1)-3*exp(x(1)-3*x(2)-0.1), ...
        9*exp(x(1)+3*x(2)-0.1)+9*exp(x(1)-3*x(2)-0.1)+0.2]);

x = [5;8];
alpha = 0.0005;
beta = 0.5;
gra = zeros(10000);
iter = zeros(10000);
for k = 1:10000
    grad1 = grad(x);
    iter(k) = k;
    gra(k) = norm(grad1);    
    
    t = 2;
    x_plus = x-alpha*t*grad1;
    
    while f(x) - f(x_plus) < -alpha*t*(grad1'*grad1)
        t = beta*t;
        x_plus = x-alpha*t*grad1;
    end
    x = x_plus;
    
    if norm(grad1)<=1e-7
        break
    end
end
disp('iteration:');
disp(k);
disp('ans=')
disp(f(x))
semilogy(iter,gra)
title('gradient descent')
xlabel('step')
ylabel('the norm of gradient of f(x)')