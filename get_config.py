import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.batch_size = 16
    config.num_workers = 8
    
    
    # hyperparameter for wgangp
    config.dropout_level = 0.05
    config.nz = 10
    config.ks = 8
    config.inc = 256
    config.outc_list = [512, 256, 128, 64]
    config.kern_list = [config.ks, 4, 4, 4]
    config.strd_list = [1, 2, 2, 2]
    config.padd_list = [0, 1, 1, 1]
    config.lrl_slope = 0.2
    
    config.beta1 = 0.5
    config.p_coeff = 10
    config.n_critic = 5
    config.clip_value = 0.01
    config.lr = 1e-5
    config.epoch_num = 64
    config.ng = 1

    config.use_cuda = False
    
    return config