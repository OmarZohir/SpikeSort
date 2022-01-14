import numpy as np 
import math
import matplotlib.pyplot as plt




#########################################################################################################



def Learning(dt,lamda,epsr,epsf,alpha, beta, mu, Nneuron,Nx, Thresh,F,C):
    ####################################################
    ###   This function  performs the learning of the
    ####  recurrent and feedforward connectivity matrices.
    ####
    ####
    ####  it takes as an argument the time step ,dt, the membrane leak, lamda, 
    ####  the learning rate of the feedforward and the recurrent
    ####  conections epsf and epsr, the scaling parameters alpha and beta of
    ####  the weights, mu the quadratic cost, the number of neurons on the
    ####  population Nneuron, the dimension of the input, the threshold of
    ####  the neurons  an the initial feedforward and recuurrent connectivity F
    ####  and C.
    ####
    ####   The output of this function are arrays, Fs abd Cs, containning the
    ####   connectivity matrices sampled at exponential time instances Fs and
    ####   Cs , The Final connectivity matrices F and C. It also gives the
    ####   Optimal decoders for each couple of recurrent and feedforward
    ####   connectivities registered in Fs and Cs. The output ErrorC contains
    ####   the distance between the current and optimal recurrent connectivity
    ####   stored in Cs. 
    ####
    ####   It also produces two figures. The first one it repsents the
    ####   connectivities before and after learning and the second figure
    ####   represents the performance of the network through learning. 
    ####
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ###############################################################################

    ##
    ##################################################################################
    ######################   Learning the optinal connectivities  ####################
    ##################################################################################
    ##################################################################################

    Nit=14000;                   #number of iteration
    Ntime=1000;                  #size of an input sequence
    TotTime=Nit*Ntime           #total time of Learning
    T= int(np.floor(np.log(TotTime)/np.log(2))); #Computing the size of the matrix where the weights are stocked on times defined on an exponential scale 
    Cs=np.zeros((T,Nneuron, Nneuron)); #the array that contains the different instances of reccurent connectivty through learning
    Fs=np.zeros((T,Nx, Nneuron));      #the array that contains the different instances of feedforward connectivty through learning
    V=np.zeros((Nneuron,)); #voltage vector of the population
    O=0;  #variable indicating the eventual  firing of a spike
    k=0;  #index of the neuron that fired
    rO=np.zeros((Nneuron,)); #vector of filtered spike train


    x=np.zeros((Nx,));   #filtered input
    Input=np.zeros((Nx,Ntime)); #raw input to the network
    Id=np.eye(Nneuron); #identity matrix

    A=2000; #Amplitude of the input
    sigma=np.absolute(30); #std of the smoothing kernel
    Gaussdist = np.arange(1,1001)-500;  #some random Gaussian distribution, starts from -499 ends at 500 with almost zero mean
    w=(1/(sigma*np.sqrt(2* np.pi)))* np.exp(-(Gaussdist*Gaussdist)/(2*sigma*sigma));#gaussian smoothing kernel used to smooth the input
    w=w/np.sum(w); # normalization oof the kernel


    j=1; # index of the (2^j)-time step (exponential times)
    l=1;
    print(f"{0} % of the learning  completed\n");

    for i in range (2,TotTime):
        if ((i/TotTime)>(l/100)):
            print(f"{l} % of the learning  completed\n");
            l=l+1;


        if (np.mod(i,np.power(2,j))==0): #registering ther weights on an exponential time scale 2^j
            Cs[j-1,:,:]=C;   #registering the recurrent weights
            Fs[j-1,:,:]=F;   #registering the Feedfoward weights
            j=j+1;

        if (np.mod(i-2,Ntime)==0): #Generating a new iput sequence every Ntime time steps 
            means = np.zeros((Nx,))
            cov = np.eye(Nx);
            #Mean has to be passed as a list not an array
            #Input  = np.transpose(np.random.multivariate_normal(meanList,cov,Ntime)); #generating a new sequence of input which a gaussion vector
            Input  = np.transpose(np.random.multivariate_normal(means,cov,Ntime)); #generating a new sequence of input which a gaussion vector

            for d in range (0,Nx-1):
                Input[d,:] = A*np.convolve(Input[d,:],w,'same'); #smoothing the previously generated white noise with the gaussian window w


        V=(1-lamda*dt)*V + dt * np.matmul(np.transpose(F),Input[:,np.mod(i,Ntime)])+ O*C[:,k]+0.001*np.random.randn(Nneuron,); #the membrane potential is a leaky integration of the feedforward input and the spikes
        x=(1-lamda*dt)*x + dt*Input[:,np.mod(i,Ntime)]; #filtered input
        breakpoint();
        
        tempArr = V - Thresh-0.01*np.random.randn(Nneuron,)-0;
        m = np.max(tempArr); #finding the neuron with largest membrane potential
        k = np.argmax(tempArr);


        if (m>=0): #if its membrane potential exceeds the threshold the neuron k spikes  
            O=1; # the spike ariable is turned to one
            F[:,k]=F[:,k]+epsf*(alpha*x-F[:,k]); #updating the feedforward weights
            C[:,k]=C[:,k] -(epsr)*(beta*(V+ mu*rO)+C[:,k]+mu*Id[:,k]);#updating the recurrent weights
            rO[k,]=rO[k,]+1; #updating the filtered spike train
        else:
            O=0;


        rO=(1-lamda*dt)*rO; #filtering the spikes

    print("learning  completed!\n");

    ##
    ##################################################################################
    ######################   Computing Optimal Decoders  #############################
    ##################################################################################
    ##################################################################################
    #####
    ##### After having learned the connectivities F and C we compute the
    ##### optimal decoding weights for each instance of the network defined by
    ##### the pairs of the FF and recurr connectivitiy matrices stocked
    ##### previously in arrays  Fs and Cs. This will allow us to compute the
    ##### decoding error over learning.
    #####
    ##################################################################################
    ##################################################################################
    ##################################################################################
    ##################################################################################
    print("Computing optimal decoders\n\n");
    TimeL=50000; # size of the sequence  of the input that will be fed to neuron
    xL=np.zeros((Nx,TimeL)); # the target output/input
    Decs=np.zeros((T,Nx,Nneuron));# array where the decoding weights for each instance of the network will be stocked
    InputL=0.3*A*np.transpose((np.random.multivariate_normal(np.zeros((Nx,)),np.eye(Nx),TimeL))); #generating a new input sequence

    for k in range (0,Nx-1):
        InputL[k,:]=np.convolve(InputL[k,:],w,'same'); #smoothing the input as before


    for t in range (1,TimeL-1):
        xL[:,t]= (1-lamda*dt)*xL[:,t-1]+ dt*InputL[:,t-1]; #compute the target output by a leaky integration of the input  



    for i in range (0,T-1):
        rOL, _ , _ = runnet(dt, lamda, np.squeeze(Fs[i,:,:]), InputL, np.squeeze(Cs[i,:,:]),Nneuron,TimeL, Thresh); # running the network with the previously generated input for the i-th instanc eof the network
        #Dec=np.transpose(np.matmul(np.transpose(rOL),np.linalg.inv(np.transpose(xL)))); # computing the optimal decoder that solves xL=Dec*rOL,  Dec=(rOL'\xL')';
        Dec = np.transpose(np.linalg.lstsq(np.transpose(rOL),np.transpose(xL))[0]);
        #Dec = np.linalg.lstsq(rOL,xL);
        Decs[i,:,:]=Dec; # stocking the decoder in Decs

    print("Optimal Decoder done")

        ##
    #################################################################################
    ###########  Computing Decoding Error, rates through Learning ###################
    #################################################################################
    #################################################################################
    #################################################################################
    #####
    ##### In this part we run the different instances of the network using a
    ##### new test input and we measure the evolution of the dedocding error
    ##### through learning using the decoders that we computed preciously. We also
    ##### measure the evolution of the mean firing rate anf the variance of the
    ##### membrane potential.
    #####
    #################################################################################
    #################################################################################
    #################################################################################
    #################################################################################

    print('Computing decoding errors and rates over learning\n')
    TimeT=10000; # size of the test input
    MeanPrate=    np.zeros((1,T));     #array of the mean rates over learning
    Error=        np.zeros((1,T));     #array of the decoding error over learning
    MembraneVar=  np.zeros((1,T));     #mean membrane potential variance over learning
    xT=           np.zeros((Nx,TimeT));#target ouput



    Trials=10; #number of trials

    for r in range (1,Trials): #for each trial
        InputT = np.transpose(A*(np.random.multivariate_normal(np.zeros((Nx,)),np.eye(Nx),TimeT))); # we genrate a new input

        for k in range (0,Nx-1):
            InputT[k,:]=np.convolve(InputT[k,:],w,'same'); # we smooth it


        for t in range (1,TimeT-1):      
            xT[:,t]= (1-lamda*dt)*xT[:,t-1]+ dt*InputT[:,t-1]; # ans we comput the target output by leaky inegration of the input       


        for i in range (1,T): #for each instance of the network
            [rOT, OT, VT] = runnet(dt, lamda, np.squeeze(Fs[i,:,:]) ,InputT, np.squeeze(Cs[i,:,:]),Nneuron,TimeT, Thresh);#we run the network with current input InputL

            xestc= np.matmul(np.squeeze(Decs[i,:,:]),rOT); #we deocode the ouptut using the optinal decoders previously computed
            Error[0,i-1]=Error[0,i-1]+np.sum(np.var(xT-xestc,ddof = 0,axis = 1))/(np.sum(np.var(xT, ddof = 0,axis = 1))*Trials);#we comput the variance of the error normalized by the variance of the target
            MeanPrate[0,i-1]=MeanPrate[0,i-1]+np.sum(np.sum(OT))/(TimeT*dt*Nneuron*Trials);#we comput the average firing rate per neuron
            MembraneVar[0,i-1]=MembraneVar[0,i-1]+np.sum(np.var(VT,ddof = 0,axis = 1))/(Nneuron*Trials);# we compute the average membrane potential variance per neuron   

            
    ##################################################################################
###########   Computing distance to  Optimal weights through Learning ############
##################################################################################
##################################################################################
###### 
###### we compute the distance between the recurrent connectivity matrics
###### ,stocked in Cs, and FF^T through learning.
######
##################################################################################
##################################################################################


    ErrorC = np.zeros((1,T));#array of distance between connectivity

    for i in range (0,T-1): #for each instance od the network
    
        CurrF=np.squeeze(Fs[i,:,:]); 
        CurrC=np.squeeze(Cs[i,:,:]); 


        Copt= -np.matmul(np.transpose(CurrF),CurrF); # we comput FF^T
        optscale = np.trace(np.matmul(CurrC.transpose(),Copt)/np.sum(np.sum(np.power(Copt,2)))); #scaling factor between the current and optimal connectivities
        Cnorm = np.sum(np.sum(np.power(Copt,2))); #norm of the actual connectivity
        ErrorC[0,i]=np.sum(np.power(np.sum((CurrC - optscale*Copt)),2))/Cnorm ;#normalized error between the current and optimal connectivity


    print("Error Calculation Done!\n")
    
    print("Started Plotting \n")
        ##
    #################################################################################
    ###########  Plotting Decoding Error, rates through Learning  ###################
    #################################################################################
    #################################################################################
    ################################################################################

    #divide into 3 vertically stacked subplots 
    lines = 3; #Number of subplots

    fig, (ax1, ax2, ax3) = plt.subplots(lines, figsize=(10,20));
    fig.subplots_adjust(hspace=0.5);

    sequence = np.linspace(1,T,T);
    xtime = dt * np.power(2,sequence);
    ax1.loglog( xtime , np.squeeze(Error) , 'k');
    ax1.set_xlabel('time');
    ax1.set_ylabel('Decoding Error');
    ax1.set_title('Evolution of the Decoding Error Through Learning')


    ax2.loglog( xtime , np.squeeze(MeanPrate), 'k');
    ax2.set_xlabel('time');
    ax2.set_ylabel('Mean Rate per neuron');
    ax2.set_title('Evolution of the Mean Population Firing Rate Through Learning ');


    ax3.loglog( xtime , np.squeeze(MembraneVar), 'k');
    ax3.set_xlabel('time');
    ax3.set_ylabel('Voltage Variance per Neuron');
    ax3.set_title('Evolution of the Variance of the Membrane Potential ');

    plt.savefig("DecError,FiringRate,Var.jpg");
    
        ##################################################################################
    ###############################  Plotting Weights  ###############################
    ##################################################################################
    ##################################################################################
    ##################################################################################




    lines=3;
    fsize=11; 

    #2 separate figures, one for the weight convergence and one for the Feedforward(F) and optimal decoder,
    #before and after learning

    ##Plotting the evolution of distance between the recurrent weights and FF^T through learning
    plt.loglog(sequence, np.squeeze(ErrorC), 'k');
    plt.xlabel('time');
    plt.ylabel('Distance to optimal weights');
    plt.title('Weight Convergence'); 

    plt.show();
    plt.savefig('WeightConvergence.jpg');


    #plotting feedforward weights before and after learning
    fig, axs = plt.subplots(lines, 2, figsize=(16,24));
    Fi= np.squeeze(Fs[0,:,:]); 
    axs[0,0].plot(Fi[0,:],Fi[1,:],'.k');  
    axs[0,0].set_xlabel('FF Weights Component 1');
    axs[0,0].set_ylabel('FF Weights Component 2');
    axs[0,0].set_title('Before Learning');
    axs[0,0].plot(0,0,'+');
    axs[0,0].set_ylim(-1,1);
    axs[0,0].set_xlim(-1,1);


    axs[0,1].plot(F[0,:],F[1,:],'.k');  
    axs[0,1].set_xlabel('FF Weights Component 1');
    axs[0,1].set_ylabel('FF Weights Component 2');
    axs[0,1].set_title('After Learning');
    axs[0,1].plot(0,0,'+');
    axs[0,1].set_ylim(-1,1);
    axs[0,1].set_xlim(-1,1);


    #scatter plot of C and FF^T Before learning
    Ci=np.squeeze(Cs[1,:,:]); 
    axs[1,0].plot(Ci, -1*np.matmul(np.transpose(Fi),Fi), '.k');
    axs[1,0].plot(0,0,'+');
    axs[1,0].set_ylim(-1,1);
    axs[1,0].set_xlim(-1,1);
    axs[1,0].set_xlabel('FF^T');
    axs[1,0].set_ylabel('Learned Rec weights');

    #scatter plot of C and FF^T After learning
    axs[1,1].plot(C, -1*np.matmul(np.transpose(F),F), '.k');
    axs[1,1].plot(0,0,'+');
    axs[1,1].set_ylim(-1,1);
    axs[1,1].set_xlim(-1,1);
    axs[1,1].set_xlabel('FF^T');
    axs[1,1].set_ylabel('Learned Rec weights');


    #scatter plot of optimal decoder and F^T before learning
    axs[2,0].plot(np.squeeze(Decs[0,:,:]),Fi,'.k');
    axs[2,0].plot(0,0,'+');
    axs[2,0].set_ylim(-1,1);
    axs[2,0].set_xlim(-1,1);
    axs[2,0].set_xlabel('Optimal decoder');
    axs[2,0].set_ylabel('F^T');

    #scatter plot of optimal decoder and F^T After learning
    axs[2,1].plot(np.squeeze(Decs[T-1,:,:]),F,'.k');
    axs[2,1].plot(0,0,'+');
    axs[2,1].set_ylim(-1,1);
    axs[2,1].set_xlim(-1,1);
    axs[2,1].set_xlabel('Optimal decoder');
    axs[2,1].set_ylabel('F^T');

    plt.savefig('FeedfowardWeightsCAndOptimalDec.jpg');
    
    print("Plotting Done\n")
    
    return Fs,Cs,F,C,Decs, ErrorC;


def runnet(dt, lamda, F ,Input, C, Nneuron, Ntime, Thresh):

##############################################################################
##############################################################################
##############################################################################
##############################################################################
####
#### This function runs the network without learning. It take as an
#### argument the time step dt, the leak of the membrane potential lamda,
#### the Input of the network, the recurrent connectivity matrix C, the feedforward
#### connectivity matrix F, the number of neurons Nneuron, the length of
#### the Input Ntime, and the Threhsold. It returns the spike trains O
#### the filterd spike trains rO, and the membrane potentials V.
####
##############################################################################
##############################################################################
##############################################################################
##############################################################################

    rO=np.zeros((Nneuron,Ntime));#filtered spike trains
    O=np.zeros((Nneuron,Ntime)); #spike trains array
    V=np.zeros((Nneuron,Ntime)); #mamebrane poterial array

    for t in range (1,Ntime-1):

        V[:,t]=(1-lamda*dt)*V[:,t-1]+dt * np.matmul(np.transpose(F),Input[:,t-1]) + np.matmul(C,O[:,t-1])+0.001*np.random.randn(Nneuron,);#the membrane potential is a leaky integration of the feedforward input and the spikes

        tempArr = V[:,t] - Thresh-0.01* np.random.randn(Nneuron,);
        m= np.max(tempArr);#finding the neuron with largest membrane potential
        k = np.argmax(tempArr);

        if (m>=0):  #if its membrane potential exceeds the threshold the neuron k spikes  
            O[k,t]=1; # the spike ariable is turned to one

        rO[:,t]=(1-lamda*dt)*rO[:,t-1]+1*O[:,t]; #filtering the spikes

    return rO, O, V; 

####################################################################################################33333
#equivalent of twoDWhite.m file

Nneuron=20; # size of the population
Nx=2;       # dimesnion of the input

lamda=50;    #membrane leak, renamed from lamda to lamda, to avoid confusion with lamda expressions
dt=0.001;     #time step

epsr=0.001;  # earning rate of the recurrent connections
epsf=0.0001; ## learning rate of the feedforward connections FF

alpha=0.18; # scaling of the Feefforward weights
beta=1/0.9;  #scaling of the recurrent weights
mu=0.02/0.9; #quadratic cost


##Initial connectivity

Fi=0.5*np.random.randn(Nx,Nneuron); #the inital feedforward weights are chosen randomely
Fi = 1*np.divide(Fi,(np.sqrt(np.ones((Nx,1))*(np.sum(np.power(Fi,2),axis = 0))))); #the FF weights are normalized
#Fi=1*(Fi./(np.sqrt(np.ones(Nx,1)*(np.sum(Fi.^2)))));  #the FF weights are normalized
Ci=-0.2*(np.random.rand(Nneuron,Nneuron))-0.5*np.eye(Nneuron); #the initial recurrent conectivity is very weak except for the autapses

Thresh=0.5; #vector of thresholds of the neurons
[Fs,Cs,F,C,Decs,ErrorC]=Learning(dt,lamda,epsr,epsf,alpha, beta, mu, Nneuron,Nx, Thresh,Fi,Ci);