t=1;
x=[1;1];
alpha = 0.001;
beta = 0.5;
iter = 1:15;
gra=zeros(15);
H = inv(hessian(x));
for k = 1:15
    d = -H*grad(x);
    while f(x) - f(x+t*d) < -alpha*t*grad(x)'*d
        t = t*beta;
    end
    
    x_plus = x + t*d;
    s = x_plus-x;
    q = grad(x_plus)-grad(x);
    w = s/(s'*q) - H*q/(q'*H*q);
    H = H + s*s'/(s'*q) - (H*(q*q')*H)/(q'*H*q) + q'*H*q*(w*w');
    x = x_plus;
    gra(k) = norm(grad(x));
    
    if norm(grad(x))<=1e-7
        break
    end
end

disp('iteration:');
disp(k);
disp('ans=')
disp(f(x))
semilogy(iter,gra)
title('BFGS')
xlabel('step')
ylabel('the norm of gradient of f(x)')