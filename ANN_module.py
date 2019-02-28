# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:52:59 2018

@author: Lucimara Bragagnolo
"""

def ann_module(input_data,output_data,reckon,num_hidden,coef,nr_epochs,val_samples,test_samples,directory,columnst,col,row,flag_train):
    
    import numpy as np
    import math
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import os
    import grass.script as gscript

    os.mkdir(directory) #Create directory

    #Scaling the data
    max_in = np.zeros((1,input_data.shape[1]))
    min_in = np.zeros((1,input_data.shape[1]))  
    max_out = 1
    min_out = 0

    
    alldata = np.concatenate((input_data,reckon))
    
    for i in range(0,input_data.shape[1]):
        max_in[0,i] = np.nanmax(alldata[:,i])
        min_in[0,i] = np.nanmin(alldata[:,i])
        
    os.chdir(directory)
    np.savetxt('max_in.txt', (max_in), delimiter=',')
    np.savetxt('min_in.txt', (min_in), delimiter=',')
    os.chdir('../')

    for j in range(0,input_data.shape[1]):
        if max_in[0,j] != 0:
            input_data[:,j] = (input_data[:,j] - min_in[0,j])/(max_in[0,j]-min_in[0,j])
    
    for j in range(0,reckon.shape[1]): 
        if max_in[0,j] != 0:
            reckon[:,j] = (reckon[:,j] - min_in[0,j])/(max_in[0,j]-min_in[0,j])

    #Resample data
    m = 1
    n = input_data.shape[0]
    n_test = int(math.ceil(test_samples*n)) #TORNAR GENERICA ESSAS %%
    n_val = int(math.ceil(val_samples*n))
    n_train = int(n - n_test - n_val)
    reg_set = np.arange(0,n) 
    T = np.zeros((n_train,1))
    V = np.zeros((n_val,1))

    flag_test = 0
    B_ind = ((n-m)*np.random.rand(2*n_test,1)+m)
    B_ind = [math.floor(x) for x in B_ind] 
        
    while flag_test == 0:
        if len(np.unique(B_ind)) == n_test:
            flag_test = 1
        else:
            del B_ind[0] 
    B_ind = np.unique(B_ind)
    B_ind = B_ind.astype(int)
    B = reg_set[B_ind]-1 #Test
    
    reg_rest = np.setdiff1d(reg_set,B) #Dataset for TRAINING and VALIDATIONS 
    
    flag_val = 0
    V_buff = ((n-n_test-m)*np.random.rand(2*n_val,1)+m)
    V_buff = [math.floor(x) for x in V_buff]
    while flag_val == 0:
        if len(np.unique(V_buff)) == n_val:
            flag_val = 1
        elif len(np.unique(V_buff)) > n_val:
            del V_buff[0]
        else:
            V_buff = ((n-n_test-m)*np.random.rand(2*n_val,1)+m)
            V_buff = [math.floor(x) for x in V_buff]
                        
    V_buff = [int(x) for x in V_buff] #Transform to integer
    V[:,0] = (reg_rest[np.unique(V_buff)])
    T[:,0] = np.setdiff1d(reg_rest,V[:,0])
    
    T = T.astype(np.int64) #Training
    V = V.astype(np.int64) #Validations
    B = B.astype(np.int64) #Test
    p = input_data.shape[1]
    input_test = np.zeros((B.shape[0],p)) 
    output_test = np.zeros((B.shape[0],1))
    input_train = np.zeros((T.shape[0],p)) 
    output_train = np.zeros((T.shape[0],1))
    input_val = np.zeros((B.shape[0],p)) 
    output_val = np.zeros((T.shape[0],1))
    
    n1 = T.shape[0]
    n2 = B.shape[0]
    n3 = V.shape[0]
            
    for i in range(0,n1-1):
        input_train[i,:] = input_data[T[i],:]
        output_train[i,:] = output_data[T[i]]
            
    for i in range(0,n2-1):
        input_test[i,:] = input_data[B[i],:]
        output_test[i,:] = output_data[B[i]]
    
    for i in range(0,n3-1):
        input_val[i,:] = input_data[V[i],:]
        output_val[i,:] = output_data[V[i]]
    
    output_data = output_data.reshape((output_data.size,1))
    
    #Validations
    def ann_validate(input_data,output_data,weights,peights,biasH,biasO,max_in,max_out,min_in,min_out):
        #Collect dimensions
        num_input = input_data.shape[1] #Number of parameters
        reg_size = input_data.shape[0] #Number of records
        num_output = output_data.shape[1]
        num_hidden = peights.shape[0] - 1 #Number of lines of weights matrix of the output layer - 1
    
        S = np.zeros((num_hidden,1))
        H = np.zeros((num_hidden,1))
        R = np.zeros((num_output,1))
        output = np.zeros((reg_size,num_output))
        erro = np.zeros((1,reg_size))
    
        def activation(x): #Sigmoid as activation function
            fx = 1/(1+math.exp(-x))
            return fx
    
        for k in range(0,reg_size):
            for i in range(0,num_hidden):
                for j in range(0,num_input):
                    S[i,0] = S[i,0] + input_data[k,j]*weights[j,i] 
                S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
                H[i,0] = activation(S[i,0])
                S[i,0] = 0
        
            for i in range(0,num_output): 
                for j in range(0,num_hidden):
                    R[i,0] = R[i,0] + H[j,0]*peights[j,i]
                R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i] 
                output[k,i] = activation(R[i,0])
                R[i,0] = 0   
    
            erro[0,k] = output[k,:] - output_data[k,:]
    
        return output,erro
    
    #Test
    def ann_test(input_data,output_data,weights,peights,biasH,biasO):
        
        #Collect dimensions
        num_input = input_data.shape[1]
        reg_size = input_data.shape[0]
        num_output = output_data.shape[1]
        num_hidden = peights.shape[0] - 1
        
        def activation(x): #activation function
            fx = 1/(1+math.exp(-x))
            return fx
            
        S = np.zeros((num_hidden,1))
        H = np.zeros((num_hidden,1))
        R = np.zeros((num_output,1))
        output = np.zeros((reg_size,num_output))
        erro_dec = np.zeros((1,reg_size))
        erro_round = np.zeros((1,reg_size))
        
        for k in range(0,reg_size):
            for i in range(0,num_hidden):
                for j in range(0,num_input):
                    S[i,0] = S[i,0] + input_data[k,j]*weights[j,i]            
                S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
                H[i,0] = activation(S[i,0])
                S[i,0] = 0
                
            for i in range(0,num_output):
                for j in range(0,num_hidden):
                    R[i,0] = R[i,0] + H[j,0]*peights[j,i]
                R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i] 
                output[k,i] = activation(R[i,0])
                R[i,0] = 0
                
            erro_dec[0,k] = output[k,:] - output_data[k,:]
            erro_round[0,k] = np.around(erro_dec[0,k])
    
        return output,erro_dec,erro_round
    
    #Reckon
    def ann_reckon(input_data,weights,peights,biasH,biasO):
        
        #Collect dimensions
        num_input = input_data.shape[1]
        reg_size = input_data.shape[0]
        num_output = 1
        num_hidden = peights.shape[0] - 1
        
        def activation(x):
            fx = 1/(1+math.exp(-x))
            return fx
            
        S = np.zeros((num_hidden,1))
        H = np.zeros((num_hidden,1))
        R = np.zeros((num_output,1))
        output = np.zeros((reg_size,num_output))
    
        for k in range(0,reg_size):
            for i in range(0,num_hidden):
                for j in range(0,num_input):
                    S[i,0] = S[i,0] + input_data[k,j]*weights[j,i]            
                S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
                H[i,0] = activation(S[i,0])
                S[i,0] = 0
                
            for i in range(0,num_output):
                for j in range(0,num_hidden):
                    R[i,0] = R[i,0] + H[j,0]*peights[j,i]
                R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i] 
                output[k,i] = activation(R[i,0])
                R[i,0] = 0
                
        output_reckon = output
        
        return output_reckon
    
    #Training
    num_input = input_train.shape[1] #Number of parameters
    reg_size = input_train.shape[0] #Number of examples
    num_output = output_train.shape[1] #Number of outputs (in this case, just 1 - susceptibility)
    weights = np.random.rand(num_input+1,num_hidden) 
    peights = np.random.rand(num_hidden+1,num_output)
    
    biasH = np.ones((num_hidden,1))
    biasO = np.ones((num_output,1))
    S = np.zeros((num_hidden,1))
    H = np.zeros((num_hidden,1))
    R = np.zeros((num_output,1))
    output = np.zeros((reg_size,num_output))
    erro = np.zeros((1,nr_epochs)) 
    erro_train = np.zeros((1,nr_epochs))
    erro_validate = np.zeros((1,nr_epochs)) 
    
    def activation(x): 
        fx = 1/(1+math.exp(-x))
        return fx
    
    for epoch in range(0,nr_epochs):
        for k in range(0,reg_size):
            for i in range(0,num_hidden):
                for j in range(0,num_input):
                    S[i,0] = S[i,0] + input_train[k,j]*weights[j,i]
                S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
                H[i,0] = activation(S[i,0])
                S[i,0] = 0
                
            for i in range(0,num_output):
                for j in range(0,num_hidden):
                    R[i,0] = R[i,0] + H[j,0]*peights[j,i]
                R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i]
                output[k,i] = activation(R[i,0])
                R[i,0] = 0
                
            #Backpropagation
            #Uptade the weights in the output layer
            for i in range(0,num_hidden):
                for j in range(0,num_output):
                    if i < (num_hidden+1):
                        peights[i,j] = peights[i,j] + coef*(output_train[k,j] - output[k,j])*output[k,j]*(1-output[k,j])*H[i,0]
                    elif i == (num_hidden+1):
                        peights[i,j] = peights[i,j] + coef*(output_train[k,j] - output[k,j])*output[k,j]*(1-output[k,j])*biasO[j,0]
                        
            #Uptade the weights in the hidden layer
            buff = 0
            for j in range(0,num_hidden):
                for i in range(0,num_input):
                    for k1 in range(0,num_output):
                        buff = buff + (output_train[k,k1] - output[k,k1])*output[k,k1]*(1-output[k,k1])*peights[j,k1]
                    
                    if i < (num_input+1):
                        weights[i,j] = weights[i,j] + coef*buff*H[j,0]*(1-H[j,0])*input_train[k,i]
                    elif i == num_input+1:
                        weights[i,j] = weights[i,j] + coef*buff*H[j,i]*(1-H[j,0])*biasH[j,0]
                        
                    buff = 0 #Zeroes the buffer variable
            
        erro_train[0,epoch] =  np.linalg.norm((output - output_train)/reg_size) 
        output_val,erro_val = ann_validate(input_val,output_val,weights,peights,biasH,biasO,max_in,max_out,min_in,min_out)
        erro_validate[0,epoch] = np.linalg.norm(erro_val)
        
        #Collects the weights in the minimum validation error
        if epoch == 0:
            W = weights
            P = peights
            epoch_min = epoch
            erro_min = 1000
        elif erro_min > np.linalg.norm(erro_validate[0,epoch]):
            W = weights
            P = peights
            epoch_min = epoch
            erro_min = np.linalg.norm(erro_validate[0,epoch])
        
    [output,erro_test,erro_round] = ann_test(input_test,output_test,weights,peights,biasH,biasO)
    
    x = np.arange(1,nr_epochs+1).reshape((nr_epochs,1))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
    plt.subplot(1,2,1)
    plt.plot(x,erro_train.T,x,erro_validate.T,'g-', epoch_min, erro_min,'g*')
    plt.xlabel('Epochs')
    plt.ylabel('Root mean square output error')
    plt.legend(('Training','Validation','Early stop'))
    plt.subplot(1,2,2)
    plt.bar(np.arange(1,(erro_test.shape[1])+1),erro_test.reshape(erro_test.shape[1]))
    plt.xlabel('Instances')
    plt.ylabel('Error (ANN output - real output)')

    os.chdir(directory)
    plt.savefig('ANN_train_val', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.2,
        frameon=None)
    os.chdir('../')
        
    rounded = np.asarray(erro_round).reshape((B.shape[0],1))
    error = sum(abs(erro_round.T))
    gscript.message(_("Test set error: "))
    gscript.message(error)
    
    #Assign the minimum weights to their original names
    weights = W
    peights = P

    os.chdir(directory)
    np.savetxt('weights.txt', (weights), delimiter=',')
    np.savetxt('peights.txt', (peights), delimiter=',')
    np.savetxt('biasH.txt', (biasH), delimiter=',')
    np.savetxt('biasO.txt', (biasO), delimiter=',')
    os.mkdir('Inputs and outputs')
    os.chdir('Inputs and outputs')
    np.savetxt('Input_test.txt', (input_test), delimiter=',')
    np.savetxt('Output_test.txt', (output_test), delimiter=',')
    np.savetxt('Input_val.txt', (input_val), delimiter=',')
    np.savetxt('Output_val.txt', (output_val), delimiter=',')
    np.savetxt('Input_train.txt', (input_train), delimiter=',')
    np.savetxt('Output_train.txt', (output_train), delimiter=',')
    np.savetxt('Error_train.txt', (erro_train), delimiter=',')
    np.savetxt('Error_val.txt', (erro_validate), delimiter=',')
    np.savetxt('Epoch_and_error_min.txt', (epoch_min,erro_min), delimiter=',')
    np.savetxt('Test_set_TOTAL_error.txt', (error), delimiter=',')
    np.savetxt('Error_test.txt', (erro_test), delimiter=',')
    param = open('ANN_Parameters.txt','w')
    param.write('Hidden neurons: '+str(num_hidden))
    param.write('\n Learning rate: '+str(coef))
    param.write('\n Epochs: '+str(nr_epochs))
    param.close()
    os.chdir('../')
    os.chdir('../')
    
    #Sensitivity analysis
    def sensitivity(input_data,output_size,weights,peights,biasH,biasO):
        
        input_size = input_data.shape[1] #Number of columns (parameters)
        npts = 200 #Number of samples in the sensitivity evaluation set
        sens_set = np.random.random_sample((npts,1)) #Return random floats in the half-open interval [0.0, 1.0)
        fixed_par_value = np.empty([input_size,1])
        ones_sens = np.ones((200,1))
        
        #Calculation of the normalized mean value of each entry
        for k in range(0,input_size): 
            fixed_par_value[k] = (np.mean(input_data[:,k]))
    
        #Pre-allocation
        input_sens = [[([1] * npts) for j in range(input_size)] for i in range(input_size)]
        output_sens = [[([0] * npts) for j in range(input_size)] for i in range(input_size)]
        input_sens = np.asarray(input_sens,dtype=float)
        
        for k1 in range(0,input_size):
            for k2 in range(0,input_size):
                if k1 == k2:
                    input_sens[k1,k2,:] = sens_set.reshape(sens_set.shape[0]).T
                else:
                    input_sens[k1,k2,:] = (ones_sens*fixed_par_value[k2]).reshape(ones_sens.shape[0])
    
        for k1 in range(0,input_size):
            input_sens2 = np.asarray(input_sens[k1]).T
            output = ann_reckon(input_sens2,weights,peights,biasH,biasO)
            output_sens[k1] = output
    
        return sens_set,fixed_par_value,input_sens,output_sens
    
    [sens_set,fixed_par_value,input_sens,output_sens] = sensitivity(input_data,output_data.shape[1],weights,peights,biasH,biasO)
    for k in range(0,input_data.shape[1]):
        plt.figure()
        plt.plot(sens_set,output_sens[k],'.')
        plt.title('Sensitivity analysis. Parameter: '+columnst[k]) 
        plt.ylabel('Output response')
        plt.xlabel('Parameter: '+columnst[k])

        os.chdir(directory)
        plt.savefig('SensitivityAnalysisVar_'+columnst[k], dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches='tight', pad_inches=0.2,
            frameon=None)
        os.chdir('../') 
    
    #Reckon
    if not flag_train:  
        output_reckon = ann_reckon(reckon,weights,peights,biasH,biasO)
    
        a = np.reshape(output_reckon,(col,row),order='F') 
        a = np.transpose(a)
    else:
        a = 0
    
    return a

#Reckon
def ann_reckon(input_data,weights,peights,biasH,biasO):
    
    import numpy as np
    import math

    #Collect dimensions
    num_input = input_data.shape[1]
    reg_size = input_data.shape[0]
    num_output = 1
    num_hidden = peights.shape[0] - 1
    
    def activation(x): 
        fx = 1/(1+math.exp(-x))
        return fx
        
    S = np.zeros((num_hidden,1))
    H = np.zeros((num_hidden,1))
    R = np.zeros((num_output,1))
    output = np.zeros((reg_size,num_output))

    for k in range(0,reg_size):
        for i in range(0,num_hidden):
            for j in range(0,num_input):
                S[i,0] = S[i,0] + input_data[k,j]*weights[j,i]            
            S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
            H[i,0] = activation(S[i,0])
            S[i,0] = 0
            
        for i in range(0,num_output):
            for j in range(0,num_hidden):
                R[i,0] = R[i,0] + H[j,0]*peights[j,i]
            R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i] 
            output[k,i] = activation(R[i,0])
            R[i,0] = 0
            
    output_reckon = output
    
    return output_reckon

def ANN_batch(input_data,output_data,reckon,hidden,trials,coef,nr_epochs,val_samples,test_samples,directory,columnst,col,row,flag_train):

    #hidden is a vector now
    #trials is the number of initial conditions
    #Train a set of neural networks and select the best one
    #Different number of hidden neurons
    #Different number of initial conditions

    import numpy as np
    import math
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import os
    import grass.script as gscript

    os.mkdir(directory) #Create directory -> REVER HERE

    #Scaling the data
    max_in = np.zeros((1,input_data.shape[1]))
    min_in = np.zeros((1,input_data.shape[1]))  
    max_out = 1
    min_out = 0

    
    alldata = np.concatenate((input_data,reckon))
    
    for i in range(0,input_data.shape[1]):
        max_in[0,i] = np.nanmax(alldata[:,i])
        min_in[0,i] = np.nanmin(alldata[:,i])
        
    os.chdir(directory)
    np.savetxt('max_in.txt', (max_in), delimiter=',')
    np.savetxt('min_in.txt', (min_in), delimiter=',')
    os.chdir('../')

    for j in range(0,input_data.shape[1]):
        if max_in[0,j] != 0:
            input_data[:,j] = (input_data[:,j] - min_in[0,j])/(max_in[0,j]-min_in[0,j])
    
    for j in range(0,reckon.shape[1]): 
        if max_in[0,j] != 0:
            reckon[:,j] = (reckon[:,j] - min_in[0,j])/(max_in[0,j]-min_in[0,j])

    #Resample data
    m = 1
    n = input_data.shape[0]
    n_test = int(math.ceil(test_samples*n)) #TORNAR GENERICA ESSAS %%
    n_val = int(math.ceil(val_samples*n))
    n_train = int(n - n_test - n_val)
    reg_set = np.arange(0,n) 
    T = np.zeros((n_train,1))
    V = np.zeros((n_val,1))

    flag_test = 0
    B_ind = ((n-m)*np.random.rand(2*n_test,1)+m)
    B_ind = [math.floor(x) for x in B_ind] 
        
    while flag_test == 0:
        if len(np.unique(B_ind)) == n_test:
            flag_test = 1
        else:
            del B_ind[0] 
    B_ind = np.unique(B_ind)
    B_ind = B_ind.astype(int)
    B = reg_set[B_ind]-1 #Test
    
    reg_rest = np.setdiff1d(reg_set,B) #Dataset for TRAINING and VALIDATIONS 
    
    flag_val = 0
    V_buff = ((n-n_test-m)*np.random.rand(2*n_val,1)+m)
    V_buff = [math.floor(x) for x in V_buff]
    while flag_val == 0:
        if len(np.unique(V_buff)) == n_val:
            flag_val = 1
        elif len(np.unique(V_buff)) > n_val:
            del V_buff[0]
        else:
            V_buff = ((n-n_test-m)*np.random.rand(2*n_val,1)+m)
            V_buff = [math.floor(x) for x in V_buff]
                        
    V_buff = [int(x) for x in V_buff] #Transform to integer
    V[:,0] = (reg_rest[np.unique(V_buff)])
    T[:,0] = np.setdiff1d(reg_rest,V[:,0])
    
    T = T.astype(np.int64) #Training
    V = V.astype(np.int64) #Validations
    B = B.astype(np.int64) #Test
    p = input_data.shape[1]
    input_test = np.zeros((B.shape[0],p)) 
    output_test = np.zeros((B.shape[0],1))
    input_train = np.zeros((T.shape[0],p)) 
    output_train = np.zeros((T.shape[0],1))
    input_val = np.zeros((B.shape[0],p)) 
    output_val = np.zeros((T.shape[0],1))
    
    n1 = T.shape[0]
    n2 = B.shape[0]
    n3 = V.shape[0]
            
    for i in range(0,n1-1):
        input_train[i,:] = input_data[T[i],:]
        output_train[i,:] = output_data[T[i]]
            
    for i in range(0,n2-1):
        input_test[i,:] = input_data[B[i],:]
        output_test[i,:] = output_data[B[i]]
    
    for i in range(0,n3-1):
        input_val[i,:] = input_data[V[i],:]
        output_val[i,:] = output_data[V[i]]
    
    output_data = output_data.reshape((output_data.size,1))
    
    def ann_train(input_train,output_train,input_val,output_val,input_test,output_test,reckon,hidden,trials,coef,nr_epochs,val_samples,test_samples,directory,columnst,col,row,flag_train):
        #Training
        num_input = input_train.shape[1] #Number of parameters
        reg_size = input_train.shape[0] #Number of examples
        num_output = output_train.shape[1] #Number of outputs (in this case, just 1 - susceptibility)
        weights = np.random.rand(num_input+1,num_hidden) 
        peights = np.random.rand(num_hidden+1,num_output)
        
        biasH = np.ones((num_hidden,1))
        biasO = np.ones((num_output,1))
        S = np.zeros((num_hidden,1))
        H = np.zeros((num_hidden,1))
        R = np.zeros((num_output,1))
        output = np.zeros((reg_size,num_output))
        erro = np.zeros((1,nr_epochs)) 
        erro_train = np.zeros((1,nr_epochs))
        erro_validate = np.zeros((1,nr_epochs)) 
        
        def activation(x): 
            fx = 1/(1+math.exp(-x))
            return fx
        
        for epoch in range(0,nr_epochs):
            for k in range(0,reg_size):
                for i in range(0,num_hidden):
                    for j in range(0,num_input):
                        S[i,0] = S[i,0] + input_train[k,j]*weights[j,i]
                    S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
                    H[i,0] = activation(S[i,0])
                    S[i,0] = 0
                    
                for i in range(0,num_output):
                    for j in range(0,num_hidden):
                        R[i,0] = R[i,0] + H[j,0]*peights[j,i]
                    R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i]
                    output[k,i] = activation(R[i,0])
                    R[i,0] = 0
                    
                #Backpropagation
                #Uptade the weights in the output layer
                for i in range(0,num_hidden):
                    for j in range(0,num_output):
                        if i < (num_hidden+1):
                            peights[i,j] = peights[i,j] + coef*(output_train[k,j] - output[k,j])*output[k,j]*(1-output[k,j])*H[i,0]
                        elif i == (num_hidden+1):
                            peights[i,j] = peights[i,j] + coef*(output_train[k,j] - output[k,j])*output[k,j]*(1-output[k,j])*biasO[j,0]
                            
                #Uptade the weights in the hidden layer
                buff = 0
                for j in range(0,num_hidden):
                    for i in range(0,num_input):
                        for k1 in range(0,num_output):
                            buff = buff + (output_train[k,k1] - output[k,k1])*output[k,k1]*(1-output[k,k1])*peights[j,k1]
                        
                        if i < (num_input+1):
                            weights[i,j] = weights[i,j] + coef*buff*H[j,0]*(1-H[j,0])*input_train[k,i]
                        elif i == num_input+1:
                            weights[i,j] = weights[i,j] + coef*buff*H[j,i]*(1-H[j,0])*biasH[j,0]
                            
                        buff = 0 #Zeroes the buffer variable
                
            erro_train[0,epoch] =  np.linalg.norm((output - output_train)/reg_size) 
            output_val,erro_val = ann_validate(input_val,output_val,weights,peights,biasH,biasO,max_in,max_out,min_in,min_out)
            erro_validate[0,epoch] = np.linalg.norm(erro_val)
            
            #Collects the weights in the minimum validation error
            if epoch == 0:
                W = weights
                P = peights
                epoch_min = epoch
                erro_min = 1000
            elif erro_min > np.linalg.norm(erro_validate[0,epoch]):
                W = weights
                P = peights
                epoch_min = epoch
                erro_min = np.linalg.norm(erro_validate[0,epoch])
            
        [output,erro_test,erro_round] = ann_test(input_test,output_test,weights,peights,biasH,biasO)
        
        error = np.linalg.norm(erro_test)
        rounded = np.asarray(erro_round).reshape((B.shape[0],1))
        error_total = sum(abs(erro_round.T))
        
        #Assign the minimum weights to their original names
        weights = W
        peights = P
        
        return output,weights,peights,biasH,biasO,erro_train,erro_validate,epoch_min,erro_min,error,erro_test,error_total     

    #Validations
    def ann_validate(input_data,output_data,weights,peights,biasH,biasO,max_in,max_out,min_in,min_out):
        #Collect dimensions
        num_input = input_data.shape[1] #Number of parameters
        reg_size = input_data.shape[0] #Number of records
        num_output = output_data.shape[1]
        num_hidden = peights.shape[0] - 1 #Number of lines of weights matrix of the output layer - 1
    
        S = np.zeros((num_hidden,1))
        H = np.zeros((num_hidden,1))
        R = np.zeros((num_output,1))
        output = np.zeros((reg_size,num_output))
        erro = np.zeros((1,reg_size))
    
        def activation(x): #Sigmoid as activation function
            fx = 1/(1+math.exp(-x))
            return fx
    
        for k in range(0,reg_size):
            for i in range(0,num_hidden):
                for j in range(0,num_input):
                    S[i,0] = S[i,0] + input_data[k,j]*weights[j,i] 
                S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
                H[i,0] = activation(S[i,0])
                S[i,0] = 0
        
            for i in range(0,num_output): 
                for j in range(0,num_hidden):
                    R[i,0] = R[i,0] + H[j,0]*peights[j,i]
                R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i] 
                output[k,i] = activation(R[i,0])
                R[i,0] = 0   
    
            erro[0,k] = output[k,:] - output_data[k,:]
    
        return output,erro
    
    #Test
    def ann_test(input_data,output_data,weights,peights,biasH,biasO):
        
        #Collect dimensions
        num_input = input_data.shape[1]
        reg_size = input_data.shape[0]
        num_output = output_data.shape[1]
        num_hidden = peights.shape[0] - 1
        
        def activation(x): #activation function
            fx = 1/(1+math.exp(-x))
            return fx
            
        S = np.zeros((num_hidden,1))
        H = np.zeros((num_hidden,1))
        R = np.zeros((num_output,1))
        output = np.zeros((reg_size,num_output))
        erro_dec = np.zeros((1,reg_size))
        erro_round = np.zeros((1,reg_size))
        
        for k in range(0,reg_size):
            for i in range(0,num_hidden):
                for j in range(0,num_input):
                    S[i,0] = S[i,0] + input_data[k,j]*weights[j,i]            
                S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
                H[i,0] = activation(S[i,0])
                S[i,0] = 0
                
            for i in range(0,num_output):
                for j in range(0,num_hidden):
                    R[i,0] = R[i,0] + H[j,0]*peights[j,i]
                R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i] 
                output[k,i] = activation(R[i,0])
                R[i,0] = 0
                
            erro_dec[0,k] = output[k,:] - output_data[k,:]
            erro_round[0,k] = np.around(erro_dec[0,k])
    
        return output,erro_dec,erro_round
    
    #Reckon
    def ann_reckon(input_data,weights,peights,biasH,biasO):
        
        #Collect dimensions
        num_input = input_data.shape[1]
        reg_size = input_data.shape[0]
        num_output = 1
        num_hidden = peights.shape[0] - 1
        
        def activation(x):
            fx = 1/(1+math.exp(-x))
            return fx
            
        S = np.zeros((num_hidden,1))
        H = np.zeros((num_hidden,1))
        R = np.zeros((num_output,1))
        output = np.zeros((reg_size,num_output))
    
        for k in range(0,reg_size):
            for i in range(0,num_hidden):
                for j in range(0,num_input):
                    S[i,0] = S[i,0] + input_data[k,j]*weights[j,i]            
                S[i,0] = S[i,0] + biasH[i,0]*weights[num_input,i]
                H[i,0] = activation(S[i,0])
                S[i,0] = 0
                
            for i in range(0,num_output):
                for j in range(0,num_hidden):
                    R[i,0] = R[i,0] + H[j,0]*peights[j,i]
                R[i,0] = R[i,0] + biasO[i,0]*peights[num_hidden,i] 
                output[k,i] = activation(R[i,0])
                R[i,0] = 0
                
        output_reckon = output
        
        return output_reckon
    
    num_hidden = len(hidden)
    erro_buff = 9999999

    for k1 in range(0,num_hidden): #hidden neurons tested
        for k2 in range(0,trials): #initial conditions

            [output,Weights,Peights,BiasH,BiasO,erro_train,erro_validate,epoch_min,erro_min,error,erro_test,error_total] = ann_train(input_train,output_train,input_val,output_val,input_test,output_test,reckon,hidden,trials,coef,nr_epochs,val_samples,test_samples,directory,columnst,col,row,flag_train)
            
            gscript.message(_("Hidden neuron: "))
            neuron_tested = hidden[k1]
            gscript.message(neuron_tested)
            gscript.message(_("Initial condition: "))
            gscript.message(k2+1)
            gscript.message(_("---------------------------------"))

            if error < erro_buff:
                erro_buff = error
                #then save the data in variables
                weights = Weights 
                peights = Peights
                biasH = BiasH
                biasO = BiasO
                Erro_train = erro_train
                Erro_val = erro_validate
                Early_stop = np.array([epoch_min,erro_min])
                Erro_test = erro_test
                Erro_test_norm = erro_buff
                neurons = hidden[k1]
                Epoch_min = epoch_min
                Erro_min = erro_min
                Total_erro = error_total

    gscript.message(_("Test set error from the best ANN: "))
    gscript.message(Total_erro)
    gscript.message(_("Best ANN hidden neurons: "))
    gscript.message(neurons)

    x = np.arange(1,nr_epochs+1).reshape((nr_epochs,1))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
    plt.subplot(1,2,1)
    plt.plot(x,Erro_train.T,x,Erro_val.T,'g-', Epoch_min, Erro_min,'g*')
    plt.xlabel('Epochs')
    plt.ylabel('Root mean square output error')
    plt.legend(('Training','Validation','Early stop'))
    plt.subplot(1,2,2)
    plt.bar(np.arange(1,(Erro_test.shape[1])+1),Erro_test.reshape(Erro_test.shape[1]))
    plt.xlabel('Instances')
    plt.ylabel('Error (ANN output - real output)')

    os.chdir(directory)
    plt.savefig('ANN_train_val', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.2,
        frameon=None)
    os.chdir('../')
    
    #Sensitivity analysis
    def sensitivity(input_data,output_size,weights,peights,biasH,biasO):
        
        input_size = input_data.shape[1] #Number of columns (parameters)
        npts = 200 #Number of samples in the sensitivity evaluation set
        sens_set = np.random.random_sample((npts,1)) #Return random floats in the half-open interval [0.0, 1.0)
        fixed_par_value = np.empty([input_size,1])
        ones_sens = np.ones((200,1))
        
        #Calculation of the normalized mean value of each entry
        for k in range(0,input_size): 
            fixed_par_value[k] = (np.mean(input_data[:,k]))
    
        #Pre-allocation
        input_sens = [[([1] * npts) for j in range(input_size)] for i in range(input_size)]
        output_sens = [[([0] * npts) for j in range(input_size)] for i in range(input_size)]
        input_sens = np.asarray(input_sens,dtype=float)
        
        for k1 in range(0,input_size):
            for k2 in range(0,input_size):
                if k1 == k2:
                    input_sens[k1,k2,:] = sens_set.reshape(sens_set.shape[0]).T
                else:
                    input_sens[k1,k2,:] = (ones_sens*fixed_par_value[k2]).reshape(ones_sens.shape[0])
    
        for k1 in range(0,input_size):
            input_sens2 = np.asarray(input_sens[k1]).T
            output = ann_reckon(input_sens2,weights,peights,biasH,biasO)
            output_sens[k1] = output
    
        return sens_set,fixed_par_value,input_sens,output_sens
    
    [sens_set,fixed_par_value,input_sens,output_sens] = sensitivity(input_data,output_data.shape[1],weights,peights,biasH,biasO)
    for k in range(0,input_data.shape[1]):
        plt.figure()
        plt.plot(sens_set,output_sens[k],'.')
        plt.title('Sensitivity analysis. Parameter: '+columnst[k]) 
        plt.ylabel('Output response')
        plt.xlabel('Parameter: '+columnst[k])

        os.chdir(directory)
        plt.savefig('SensitivityAnalysisVar_'+columnst[k], dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches='tight', pad_inches=0.2,
            frameon=None)
        os.chdir('../') 
    
    os.chdir(directory)
    np.savetxt('weights.txt', (weights), delimiter=',')
    np.savetxt('peights.txt', (peights), delimiter=',')
    np.savetxt('biasH.txt', (biasH), delimiter=',')
    np.savetxt('biasO.txt', (biasO), delimiter=',')
    os.mkdir('Inputs and outputs')
    os.chdir('Inputs and outputs')
    np.savetxt('Input_test.txt', (input_test), delimiter=',')
    np.savetxt('Output_test.txt', (output_test), delimiter=',')
    np.savetxt('Input_val.txt', (input_val), delimiter=',')
    np.savetxt('Output_val.txt', (output_val), delimiter=',')
    np.savetxt('Input_train.txt', (input_train), delimiter=',')
    np.savetxt('Output_train.txt', (output_train), delimiter=',')
    np.savetxt('Error_train.txt', (Erro_train), delimiter=',')
    np.savetxt('Error_val.txt', (Erro_val), delimiter=',')
    np.savetxt('Epoch_and_error_min.txt', (Early_stop), delimiter=',')
    np.savetxt('Test_set_TOTAL_error.txt', (Total_erro), delimiter=',')
    np.savetxt('Error_test.txt', (Erro_test), delimiter=',')
    param = open('ANN_Parameters.txt','w')
    param.write('Hidden neurons of best ANN: '+str(neurons))
    param.write('\n Learning rate: '+str(coef))
    param.write('\n Epochs: '+str(nr_epochs))
    param.write('\n Hidden neurons tested: '+str(hidden))
    param.write('\n Number of initial conditions tested: '+str(trials))
    param.close()
    os.chdir('../')
    os.chdir('../')

    #Reckon
    if not flag_train:  
        output_reckon = ann_reckon(reckon,weights,peights,biasH,biasO)
    
        a = np.reshape(output_reckon,(col,row),order='F') 
        a = np.transpose(a)
    else:
        a = 0
    
    return a