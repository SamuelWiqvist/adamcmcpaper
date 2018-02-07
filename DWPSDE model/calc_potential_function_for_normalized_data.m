
syms x c d A f g power1 power2 m  


% potential 
V(x,c,d,A,f,g,power1,power2) = abs(( abs(x).^(power1)/2 - d + g.*x)).^(power2)/2 + 1/2*(A*x.^2)

V_prime = diff(V, x)