function [ecg] =  multichannel_ecg(num_channels, num_samples)
    if nargin < 2
        num_samples = 2000;
    end
    
    ecg = zeros(num_channels, num_samples);
    for i=1:num_channels
        ecg(i,:) = gen_rand_ecg(num_samples); 
    end
    
end