function [ecg] =  gen_rand_ecg(num_samples)

if (nargin < 1)
    num_samples = 2000;
end

dt = 0.001; %Time step for the ecg signals
x=dt:dt:num_samples*dt;

%possible heart rates (Expected value = 9.5*8= 76)
rate = gamrnd(9.5,8);
if (rate < 65) %Do not allow heart rates less than 65 (make it very inprobable)
    rate = random('Normal', 70, 3);
end

li=30/rate;

%p wave specifications
% Default values
a_pwav=0.25;
d_pwav=0.09;
t_pwav=0.16;

% set the parameters to random variables
a_pwav= random('Normal', a_pwav, 0.2*a_pwav); 
d_pwav= random('Normal', d_pwav, 0.2*d_pwav); 
t_pwav= random('Normal', t_pwav, 0.2*t_pwav); 


%q wave specifications
%Default values
    a_qwav=0.025;
    d_qwav=0.066;
    t_qwav=0.166;
    
%set the parameters to random variables with the mean being the default
%value, and the standarad deviation is 20% of the mean value
   a_qwav= random('Normal', a_qwav, 0.2*a_qwav);
   d_qwav= random('Normal', d_qwav, 0.2*d_qwav);


%qrs wave specifications
%Default values 
    a_qrswav=1.6;
    d_qrswav=0.11;

%set the parameters to random variables
   a_qrswav= random('Normal', a_qrswav, 0.2*a_qrswav);
   d_qrswav= random('Normal', d_qrswav, 0.2*d_qrswav);


%s wave specifications
%Default values
    a_swav=0.25;
    d_swav=0.066;
    t_swav=0.09;
%set the parameters to random variables
   a_swav= random('Normal', a_swav, 0.2*a_swav);
   d_swav= random('Normal', d_swav, 0.2*d_swav);


%t wave specifications
%Default values
    a_twav=0.35;
    d_twav=0.142;
    t_twav=0.2;
%set the parameters to random variables
   a_twav=random('Normal', a_twav, 0.2*a_twav);
   d_twav=random('Normal', d_twav, 0.2*d_twav);
   t_twav=random('Normal', t_twav, 0.2*t_twav);

%u wave specifications
%Default values
    a_uwav=0.035;
    d_uwav=0.0476;
    t_uwav=0.433;
    
%set the parameters to random variables
   a_uwav=random('Normal', a_uwav, 0.2*a_uwav);
   d_uwav=random('Normal', d_uwav, 0.2*d_uwav);
 
 %generate p signal
 pwav=p_wav(x,a_pwav,d_pwav,t_pwav,li);

 
 %qwav output
 qwav=q_wav(x,a_qwav,d_qwav,t_qwav,li);

    
 %qrswav output
 qrswav=qrs_wav(x,a_qrswav,d_qrswav,li);

 %swav output
 swav=s_wav(x,a_swav,d_swav,t_swav,li);

 
 %twav output
 twav=t_wav(x,a_twav,d_twav,t_twav,li);

 
 %uwav output
 uwav=u_wav(x,a_uwav,d_uwav,t_uwav,li);

 %ecg output
 ecg=pwav+qrswav+twav+swav+qwav+uwav;
end