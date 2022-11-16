% Homework 3: Short-Time Time Domain Features
% Using Hamming window, different time-domain features are computed
% and applied to TIMIT speech sample.
% Myungeun Lee
% 11/08/2022

function homework3

% 1. Implement a Hamming window of length L, L=401
L = 401;
h = zeros(L);
for n = 1:L
    h(n) = 0.54 - 0.46*cos(2*pi*(n-1)/(L-1));
end;
plot(h);
title('Hamming Window');
pause;
H = fft(h);

subplot(2,1,1), plot(abs(H));
title('FFT of Hamming Window');
subplot(2,1,2), plot(angle(H));
pause;

% 2. Short0time energy applied to TIMIT sample
% Implement Hamming windows of length L, L =51, 101, 201, and 401

% Read TIMIT speech
fname = 'LDC93s1.wav';
x = audioread(fname);

L = [51,101,201,401];
% Compute Energy Function for different window length.
for k=1:4,
    w = hamming(L(k));
    % compute modified window
    h = w.^2;
    y = conv(x.^2,h);
    N = length(y);

    % subsample every 4 samples
    for m=1:floor(N/4),
        energy(m) = y((m-1)*4+1);
    end;
    subplot(2,1,1), plot(x);
    title('Original Speech');
    subplot(2,1,2), plot(energy);
    title('Energy, window size L=')
    pause;
end;


% 3. Compute zero crossings
for k = 1:4,
    w = hamming(L(k));
    h = w.^2;
    N = length(x)-1;
    diff = zeros(1,N);
    for n = 1:N,
        diff(n) = abs(sign(x(n+1))-sign(x(n)));
    end;
    y = conv(diff,w);
    M = length(y);
    for m=1:floor(M/4),
        zc(m) = y((m-1)*4+1);
    end;
    subplot(2,1,1), plot(x);
    title('Original Speech');
    subplot(2,1,2), plot(zc);
    title('Zero Crossing, window size L=')
    pause;
end;

% 4. Compute short-time correlation
rk = zeros(1,250);
for l = 1:4,
    w = hamming(L(l));
    N = length(x)-640;
    % iterate for 640 (40ms) samples
    M = floor(N/320);
    corr = zeros(M,250);
    for i = 1:320:N,
        % iterate for lags ranging from 0 to 249
        for k = 0:249,
            x1 = x(i:i+320);
            y1 = conv(x1,w);
            x2 = x(i+k:i+k+320);
            y2 = conv(x2,w);
            rk(k+1) = y1'*y2;
        end;
        corr((i-1)/320+1,:) = rk;
    end;    
    subplot(2,1,1); plot(x);
    title('Original Speech');
    subplot(2,1,2); imagesc(corr');
    title('Correlation, window size L=');
    pause;
    waterfall(corr(20:30,:));
    title('Waterfall, window size L=');
    pause;
end;
            





