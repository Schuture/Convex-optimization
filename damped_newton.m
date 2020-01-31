x = [5;8];
alpha = 0.001;
beta = 0.5;
iter = 1:50;
gra = zeros(50);
for k = 1:50
    grad1 = grad(x);
    grad2 = hessian(x);
    d = -inv(grad2)*grad1;
    gra(k) = norm(grad1);
    
    t = 1;
    while f(x) - f(x+t*d) < -alpha*t*grad1'*d
        t = beta*t;
    end
    x = x+t*d;
    if norm(grad1)<=1e-7
        break
    end
end
ret = f(x);
disp('iteration:');
disp(k);
disp('ans=')
disp(f(x))
semilogy(iter,gra)
title('damped newton')
xlabel('step')
ylabel('the norm of gradient of f(x)')