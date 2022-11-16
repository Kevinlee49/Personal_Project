% Homework 2: Read TIMIT speech data into a vector x and divide into
% small segments, then plot. Different speech regions, silence, voiced,
% and unvoiced periods, are manually identified from the speech waveform.
% It will be further divided into overlapping window of 20 ms long, and
% apply FFT to plot spectogram.
% Myungeun Lee
% 10/11/2022

function homework2

% Write a matlab code to read the wave file into a vector x.
x=audioread("LDC93S1.wav");

N = length(x);

% Plot the speech vector x
plot(x);
pause;

% Divide the vector x into five segments and plot them.
M = ceil(N/5);
M1 = floor(N/5);

for ns = 1:5
    if(ns<5)
        y = x((ns-1)*M+1:ns*M);
    else % last segment
        y = x((4*M)+1:N);
    end
    plot(y)
    pause;
end

% Manually identify silence, voiced and unvoiced speech.

% Write a matlab code to play the vector x (sampling frequency = 16 KHz).
Fs = 16*10^3;
sound(x,Fs);

% Divide the speech file into 20ms segments (320 samples each), and
% compute spectogram then display it

M=floor(N/2)-160;
S=zeros(M,160);
for i=1:M
    is = (i-1)*2+1;
    y = x(is:is+319);
    Y = fft(y);
    z = abs(Y);
    S(i,:) = z(1:160);
  
end;
S = S';
%colormap(gray);
%imagesc(S);
imagesc(log(S(1:160,:) + ones(160,M)));
axis xy;

