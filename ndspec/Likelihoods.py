import numpy as np
from .SimpleFit import SimpleFit as simple

def delchi(data,err,model,noise=None,noise_err=None,residuals=False):

    #check that both noise and noise_err are present, or absent
    if (noise is None and noise_err is not None):
        raise ValueError("The background is not defined but its error is")
    if (noise is not None and noise_err is None):
        raise ValueError("The background is defined but its error is not")
    
    if noise_err is not None:
        err = np.sqrt(np.power(err,2)+np.power(noise_err,2))
        delchi = (data-noise-model)/err
    else:
        delchi = (data-model)/err
    
    if residuals is False:
        return np.sum(delchi)    
    elif residuals is True:
        return delchi
    
def ratio(data,err,model,noise=None,noise_err=None,residuals=True,bars=True):

    if noise is not None:
        data = data-noise 
    if noise_err is not None:
        err = np.sqrt(np.power(err,2)+np.power(noise_err,2))

    ratio = data/model 

    if bars is True:
        bars = err/model 
        return ratio, bars
    else: 
        return ratio

def cstat(data,model,exp,widths,noise=None,residuals=False):
       
    #fix the factor needed to convert to integrated counts, then calculate 
    #the model and data in integrated counts  
    conv_factor = exp*widths
    data = data*conv_factor
    model = model*conv_factor

    if noise is None:   
        cstat = model - data + data*(np.log(data)-np.log(model))        
    else:
        #convert the background array to counts 
        noise = noise*conv_factor       
        #see https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node340.html
        #if the bkg and source exposures are identical, our model array is
        #== ts*mi from the xspec docs which simplifies things
        #first we calculate the factor fi in the xspec docs
        cstat = np.zeros(len(data))
        bkg_num = np.zeros(len(data))
        bkg_den = np.zeros(len(data))
        
        test_sign = 2.*model - data - noise 
        #define the bkg model - fi in the xspec docs
        d_factor = np.sqrt(np.power(test_sign,2.)+8.*model*noise)   
        
        mask = (test_sign>=0)        
        bkg_num[mask] = 2.*noise[mask]*model[mask]/(exp)
        bkg_den[mask] = test_sign[mask] + d_factor[mask]    
        bkg_num[~mask] = d_factor[~mask] - test_sign[~mask] 
        bkg_den[~mask] = 2.*exp             
        bkg_model = bkg_num/bkg_den*exp 
        
        #handle the exception of data being 0
        mask = (data==0)
        mask_model = model[mask]
        mask_noise = noise[mask]
        cstat[mask] = mask_model - mask_noise*np.log(0.5)

        #handle the exceptionof noise being 0 and low counts
        mask = (noise==0) & (test_sign<0)
        mask_model = model[mask]
        mask_data = data[mask] 
        cstat[mask] = -mask_model - mask_data*np.log(0.5)
        
        #handle the exception of noise being 0 and high counts 
        mask = (noise==0) & (test_sign>=0) 
        mask_model = model[mask]
        mask_data = data[mask] 
        cstat[mask] = (mask_model - mask_data + 
                      mask_data*(np.log(mask_data) - np.log(mask_model)))
       
        #handle the bins without exceptions
        mask = (data != 0) & (noise !=0)
        mask_model = model[mask]
        mask_data = data[mask]
        mask_noise = noise[mask]
        mask_bkg_model = bkg_model[mask]                
        cstat[mask] = (mask_model + 2.*mask_bkg_model - 
                       mask_data*np.log(mask_model+mask_bkg_model) -
                       mask_noise*np.log(mask_bkg_model) - 
                       mask_data*(1.-np.log(mask_data)) - 
                       mask_noise*(1.-np.log(mask_noise)))
    
    if residuals is False:
        return np.sum(2.*cstat)    
    elif residuals is True:
        return 2.*cstat
