function homework4

% 1. Generate 1024-point white Gaussian random noise w, with zero mean and unit variance. 
N = 1024;
p = 20;

w = randn(N,1);
x = zeros(N,1);

% 2. Generate the 1024-point signal x defined by the following equation
for n=1:N,
    x(n) = cos(0.1*pi*n) + 0.2*cos(0.15*pi*n) + 0.2*w(n);
end;

% 3. Estimate the parameters of 20th order autoregressive (linear prediction) model by a direct inversion method.
% ap ... a1
% a = inv(zz)*zy;
% a = flip(a);
% a = [1; -a];
% plot(a)
% title('A direct inversion method')
% pause;


% 3. Estimate the parameters of 20th order autoregressive (linear
% prediction) model by Burg-Anderson method (arburg).
zz = zeros(p);
zy = zeros(p,1);
for n=p+1:N,
    z = x(n-p:n-1);
    zz = zz + z*z';
    zy = zy + z*x(n);
end;

% a = aryule(x,p);
a = arburg(x,p);
a = a';
plot(a);
title('Burg-Anderson method(arburg)')
pause;


% 4. Compute the power spectral density using the estimated parameters and plot the spectrum.
% For each frequency, power spectrum is calculated
phi = zeros(p+1,1);
S = zeros(N,1);
for n=1:N,
    omega = (n-1)*pi/N;
    for m=1:p+1,
        phi(m) = exp(j*(m-1)*omega);
    end;
    S(n) = 1/abs(a'*phi)^2;
end;
plot(log(S));
title('Spectrum by a Burg-Anderson method(arburg)')
pause;


% 5. Find the roots of the autoregressive polynomial and find the spectral peaks from the computed roots.  
% Calculate roots of AR polynomial
r = roots(a);
plot(r,'x');
title('autoregressive polynomial by Burg-Anderson method(arburg)')
pause;

% Find 2 pairs f dominant poles
[r1,I] = sort(1-abs(r));
%for n=1:4,
%    r(I(n))
%end;
% Display two dominant spectral peaks
omega1 = angle(r(I(1)))
omega2 = angle(r(I(3)))

