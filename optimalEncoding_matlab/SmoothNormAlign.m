function [RecAligned, InputAligned, RMSE] = SmoothNormAlign(reconstruction, Input, Nx, Time)
    %Moving average of reconstructed signal
    %sliding window of 10 signals
    RecSmoothened(1,:) = movmean(reconstruction(1,:), 10);
    RecSmoothened(2,:) = movmean(reconstruction(2,:), 10);
 
    %Normalizing both signals
    for i = 1:Nx
        RecNormalized(i,:) = RecSmoothened(i,:)/max(RecSmoothened(i,:));
        InputLNorm(i,:) = Input(i,:)/max(Input(i,:));
    end

    %Aligning the Normalized Input Signal and the reconstructed signal
    %needs the signal processing toolbox
    %Align each dimension on its own

    for i = 1:size(Input,1)
        [RecTemp(i,:), InputTemp(i,:),D] = alignsignals(RecNormalized(i,:), InputLNorm(i,:));    
    end
    
    %%The Input shall be delayed by D samples, discard the first 5*D samples in
    %%both, and discard the last D samples in InputTemp
    InputAligned = zeros(Nx,Time - 5*D);
    RecAligned = zeros(Nx,Time - 5*D);
    startpoint = abs(5*D)+1;
    endpoint = Time;
    InputAligned = InputTemp(:,startpoint:endpoint);
    RecAligned = RecTemp(:,startpoint:endpoint);
    for i = 1:Nx
        RMSE(i,:) = sqrt(mean((InputAligned(i,:) - RecAligned(i,:)).^2));
    end
end