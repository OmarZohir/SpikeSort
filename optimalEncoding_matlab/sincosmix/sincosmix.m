function [W,A] = sincosmix(t, n_max, n_min, f_min, f_max)
%SINCOSMIX Summary of this function goes here
%%%
%      Supperposition of sines with integer frequencies.
%      t -> time vector (should be vertical and 2D).
%      n_max -> maximum integer for frequencies
%      n_min -> minimum integer for frequencies
%      f_min, f_max -> min and max frequency ranges from which signal will
%      be generated
%   

%Assign values for n_min and n_max, if not provided
if nargin < 3
    n_max = 50; n_min = 1; 
    f_min = 0; f_max = 40000;
else if nargin < 5
    f_min = 0; f_max = 40000;
end


%n_max and n_min has to be defined by that point, N is the number of
%frequencies the signal shall include
N = randi([n_min n_max], 1);
%Assign random amplitudes/weights for each of those frequencies, in both
%sin and cosine component
A = randn(2*N,1);

% Divide the range f_min:f_max into an array of length equal to 5*N,
%Then take N random samples from the array of frequencies of length 5*N,
%without replacement
f_range = linspace(f_max,f_min,5*N);
f = randsample(f_range, N);

omega = 2*pi*f;
omeg2 = omega.^2;

T = t(end);

s = sin (omega .* t) ./ omega;
c = cos (omega .* t) ./ omega;

%Remove the mean value
avgW = ( 1 - cos(omega .*T) ) ./ omeg2;
W   = s - avgW;
avgW = sin (omega .* T) ./ omeg2;
W = horzcat(W,c-avgW);
W = W * A;

%Scale W to [0 1]
W  = (W - min(W)) ./ (max(W) - min(W));
end

