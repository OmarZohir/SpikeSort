function [RecAligned, InputAligned, RMSE] = SmoothNormAlign(reconstruction, Input, Nx, Time)
    %Moving average of reconstructed signal
    %sliding window of 10 signals
    RecSmoothened(1,:) = movmean(reconstruction(1,:), 10);
    RecSmoothened(2,:) = movmean(reconstruction(2,:), 10);
%     RecSmoothened = reconstruction;
 
    %Normalizing both signals
    for i = 1:Nx
        %Normalize Reconstructed signal
        if min(RecSmoothened(i,:)) > 0
            signRec = 0;
        else
            signRec = 1;
        end
        RecNormalized(i,:) = RecSmoothened(i,:) - (-1) ^ signRec * abs(min(RecSmoothened(i,:)));
        RecNormalized(i,:) = RecNormalized(i,:)/max(RecNormalized(i,:));
        
        %Normalize Input Signal
        if min(Input(i,:)) > 0
            signInput = 0;
        else
            signInput = 1;
        end
        InputLNorm(i,:) = Input(i,:) - (-1) ^ signInput * abs(min(Input(i,:)));
        InputLNorm(i,:) = InputLNorm(i,:)/max(InputLNorm(i,:));
    end

    %Aligning the Normalized Input Signal and the reconstructed signal
    %needs the signal processing toolbox
    %Align each dimension on its own
     %For Smoothened input, the delay on both dimensions shall be the same
     %for i=1:size(InputLNorm,1)
     %    [RecTemp(i,:), InputTemp(i,:),D] = alignsignals(RecNormalized(i,:), InputLNorm(i,:));
     %end
    
    %For Non-smoothened input, delay only a single dimension, then apply
    %the same delay to the second (other) dimension(s)
     [RecTemp(1,:), InputTemp(1,:),D] = alignsignals(RecNormalized(1,:), InputLNorm(1,:));
     for i= 2:size(InputLNorm,1)
         InputTemp(i,:) = [zeros(1,abs(D)) InputLNorm(i,:)]; 
         RecTemp(i,:) = RecNormalized(i,:);
     end
    
    
    %%The Input shall be delayed by D samples, discard the first 5*D samples in
    %%both, and discard the last D samples in InputTemp
    %InputAligned = zeros(Nx,Time - 5*abs(D));
    %RecAligned = zeros(Nx,Time - 5*abs(D));
    %an estimate of size of abs(D) being around 20 samples
    InputAligned = zeros(Nx,Time - 100);
    RecAligned = zeros(Nx,Time - 100);
    startpoint = abs(100)+1;
    endpoint = Time;
    InputAligned = InputTemp(:,startpoint:endpoint);
    RecAligned = RecTemp(:,startpoint:endpoint);
    for i = 1:Nx
        Error(i,:) = sqrt(mean((InputAligned(i,:) - RecAligned(i,:)).^2));
    end
    RMSE = Error(1,:);
end