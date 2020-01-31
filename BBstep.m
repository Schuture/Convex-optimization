x=[1;0.1];
alpha = 0.3;
beta = 0.5;
iter = 1:1000;
gra=zeros(1000);
t = 1;
for k = 1:1000
    d = -alpha*grad(x);
    x_plus = x + t*d;
    s = x_plus-x;
    y = grad(x_plus)-grad(x);
    %two updates of t
    %t = s'*y/(y'*y); 
    t = s'*s/(s'*y);
    while f(x) - f(x+t*d) < -alpha*t*grad(x)'*d
        t = t*beta;
    end

    x = x+t*d;
    gra(k) = norm(grad(x));
    
    if norm(grad(x))<=1e-7
        break
    end
end

disp('ans=')
disp(f(x))
semilogy(iter,gra)
title('BB step')
xlabel('step')
ylabel('the norm of gradient of f(x)')