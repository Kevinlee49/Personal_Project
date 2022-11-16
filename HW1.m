%
% Homework 1: Generate a signal with two sinusoid and corrupted by additive with Gaussian noise, then
% apply DFT to see spectrum of the signal. We repeat for different frequencies
% 
% 

function homework1
N = 1024;
% 1. Generate 1024-point white Gaussian random noise w, with zero mean and unit variance.
w = randn(N,1);

% Plot w
subplot(4,1,1);
plot(w);
grid on
title('plot w')

% Create a 1024-element column vector
x = zeros(N,1);

% 2. Generate the 1024-point signal x defined by the following equation.
for n=1:N
    x(n) = cos(0.1*pi*n) + 0.2*sin(0.2*pi*n) + 0.2*w(n);
end
subplot(4,1,2);
plot(x); % 3. plot(x)
grid on
title('plot x')
% 4. Compute the DFT of x(n), and obtain X(k).
X = fft(x);

% Plot real part of X
%plot(real(X));
%pause;

% Plot imaginary part of X
%plot(imag(X));
%pause;

% Plot magnitude of X
%plot(abs(X));
%pause;

% Plot phase of X
%plot(angle(X));
%pause;

% 5. Plot the periodogram.
Px = abs(X).^2;
Px = Px(1:512);

% Plot Px
subplot(4,1,3);
plot(Px) % there are two peaks, 1st peak x,y = (52,226187), 2nd peak x,y = (103,6764)
grid on
title('plot Px : periodogram')
% 6. Repeat the above experiment by changing the frequency of the second sinusoid.  Do you observe any differences?

N = 1024;
w = randn(N,1);
x = zeros(N,1);

for n=1:N
    x(n) = cos(0.1*pi*n) + 0.2*sin(0.12*pi*n) + 0.2*w(n);
end
subplot(4,1,4);
plot(x);
grid on
title('6th sinusoid')
X = fft(x)






