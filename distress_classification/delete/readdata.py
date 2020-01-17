from tensorflow.keras import utils
import numpy as np

def gapv164(tr = 3, val = 1, te = 1):
    num_classes = 2
    # just intact and crack
    from importlib.machinery import SourceFileLoader
    MODULENAME = "laodgaps"
    MODULEPATH = "/home/ali/my_project/large_files/gaps/loadgaps.py"
    lgaps = SourceFileLoader(MODULENAME, MODULEPATH).load_module()
    
    
   
    # data is in the format of float 32 and dimenstion of (1,64,64)
    gaps_v1_tr_data, gaps_v1_tr_label, gaps_v1_va_data, \
    gaps_v1_va_label , gaps_v1_te_data, gaps_v1_te_label = lgaps.loadv1(
                                                            v1_tr_patch_num = tr, # upto 154
                                                            v1_va_patch_num = val, # upto 7
                                                            v1_te_patch_num = te # upto 26)
                                                            )
    
    # 3 channel Convert data from shape of (:,1,64, 64) to (:, 64, 64, 3) 
    gaps_v1_tr_data_3ch = np.stack((gaps_v1_tr_data[:,0,:,:],)*3, axis=-1)
    gaps_v1_va_data_3ch = np.stack((gaps_v1_va_data[:,0,:,:],)*3, axis=-1)
    gaps_v1_te_data_3ch = np.stack((gaps_v1_te_data[:,0,:,:],)*3, axis=-1)
    
    # Convert class arrays to binary class matrices.
    gaps_v1_tr_label_binary = utils.to_categorical(gaps_v1_tr_label, num_classes)
    gaps_v1_va_label_binary = utils.to_categorical(gaps_v1_va_label, num_classes)
    gaps_v1_te_label_binary = utils.to_categorical(gaps_v1_te_label, num_classes)
    
    data_name = 'gaps_v1_64'
    #x_train  = gaps_v1_tr_data 
    #y_train  = gaps_v1_tr_label
    #x_valid  = gaps_v1_va_data
    #y_valid  = gaps_v1_va_label
    #x_test = gaps_v1_te_data
    #y_test = gaps_v1_te_label
    print('data name  is: ' + data_name)
    print('x_train shape:', gaps_v1_tr_data_3ch.shape)
    print(gaps_v1_tr_data_3ch.shape[0], 'train samples')
    print(gaps_v1_va_data_3ch.shape[0], 'valid samples')
    print(gaps_v1_te_data_3ch.shape[0],  'test samples')

    
    return gaps_v1_tr_data_3ch, gaps_v1_tr_label_binary, gaps_v1_va_data_3ch, \
    gaps_v1_va_label_binary , gaps_v1_te_data_3ch, gaps_v1_te_label_binary, data_name
  
    
#    return gaps_v1_tr_data, gaps_v1_tr_label_binary, gaps_v1_va_data, \
#    gaps_v1_va_label_binary , gaps_v1_te_data, gaps_v1_te_label_binary, #data_name
