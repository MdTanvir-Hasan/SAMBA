@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    seq_in: int
    seq_out: int
    d_state: int =128
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 3
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two PyTorch tensors.

    Args:
    x (torch.Tensor): First input tensor.
    y (torch.Tensor): Second input tensor.

    Returns:
    torch.Tensor: Pearson correlation coefficient.
    """
    # Ensure the tensors are of type float32
    x = x.float()
    y = y.float()

    # Compute the mean of each tensor
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)

    # Compute the deviations from the mean
    dev_x = x - mean_x
    dev_y = y - mean_y

    # Compute the covariance between x and y
    covariance = torch.sum(dev_x * dev_y)

    # Compute the standard deviations of x and y
    std_x = torch.sqrt(torch.sum(dev_x ** 2))
    std_y = torch.sqrt(torch.sum(dev_y ** 2))

    # Compute the Pearson correlation coefficient
    pearson_corr = covariance / (std_x * std_y)

    return pearson_corr

def rank_tensor(x):
    """
    Return the ranks of elements in a tensor.
    
    Args:
    x (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Ranks of the input tensor elements.
    """
    # Get the sorted indices
    sorted_indices = torch.argsort(x)
    
    # Create an empty tensor to hold the ranks
    ranks = torch.zeros_like(sorted_indices, dtype=torch.float)
    
    # Assign ranks based on sorted indices
    ranks[sorted_indices] = torch.arange(1, len(x) + 1).float()
    
    return ranks

def rank_information_coefficient(x, y):
    """
    Calculate the Rank Information Coefficient (RIC) or Spearman's Rank Correlation Coefficient.
    
    Args:
    x (torch.Tensor): First input tensor.
    y (torch.Tensor): Second input tensor.

    Returns:
    torch.Tensor: Rank Information Coefficient (RIC).
    """
    # Get the ranks of the elements in x and y
    rank_x = rank_tensor(x)
    rank_y = rank_tensor(y)

    # Calculate the mean rank for both tensors
    mean_rank_x = torch.mean(rank_x)
    mean_rank_y = torch.mean(rank_y)

    # Calculate the covariance of the rank variables
    covariance = torch.sum((rank_x - mean_rank_x) * (rank_y - mean_rank_y))

    # Calculate the standard deviations of the ranks
    std_rank_x = torch.sqrt(torch.sum((rank_x - mean_rank_x) ** 2))
    std_rank_y = torch.sqrt(torch.sum((rank_y - mean_rank_y) ** 2))

    # Calculate the Spearman rank correlation (RIC)
    ric = covariance / (std_rank_x * std_rank_y)
    
    return ric

for i in range(5):

    Mode = 'train'
    DEBUG = 'True'
    DATASET = 'PEMSD8'      #PEMSD4 or PEMSD8
    DEVICE = 'cuda:0'
    MODEL = 'AGCRN'

#get configuration
    config_file = './{}_{}.conf'.format(DATASET, MODEL)
#print('Read configuration file: %s' % (config_file))
    config = configparser.ConfigParser()
    config.read(config_file)




#parser

    args={"dataset":DATASET,"mode":Mode,"device":DEVICE,"debug":DEBUG,"model":MODEL,"cuda":True,"val_ratio":0.15,"test_ratio":0.15,
      "lag":window,"horizon":predict,"num_nodes":XX.shape[2],"tod":False,"normalizer":'std',"column_wise":False,"default_graph":True,
     "input_dim":1,"output_dim":1,"embed_dim":10,"rnn_units":128,"num_layers":3,"cheb_k":3,"loss_func":'mae',"seed":1,
     "batch_size":32,"epochs":1100,"lr_init":0.001,"lr_decay":True,"lr_decay_rate":0.5,"lr_decay_step":[40,70,100],
      "early_stop":True,"early_stop_patience":200,"grad_norm":False,"max_grad_norm":5,"real_value":False,"mae_thresh":None,
      "mape_thresh":0,"log_dir":'./',"log_step":20,"plot":False,"teacher_forcing":False,"d_in":32,"hid":32}




#init model
    model = SAMBA(ModelArgs(args.get("d_in"),args.get("num_layers"),args.get("num_nodes"),args.get('lag'),args.get('horizon')),args.get('hid'),args.get('lag'),args.get('horizon'),args.get('embed_dim'),args.get("cheb_k"))
    model = model.cuda()
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    if args.get('loss_func') == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif args.get('loss_func') == 'mae':
        loss = torch.nn.L1Loss().to(args.get('device'))
    elif args.get('loss_func') == 'mse':
        loss = torch.nn.MSELoss().to(args.get('device'))
    else:
        raise ValueError

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.get('lr_init'), eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
    lr_scheduler = None
    if args.get('lr_decay'):
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in args.get('lr_decay_step')]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
        milestones=[0.5 * args.get('epochs'),0.7 * args.get('epochs'), 0.9 * args.get('epochs')],gamma=0.1)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)



#start training
    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, args=args, lr_scheduler=lr_scheduler)

    trainer.train()

    y1,y2=trainer.test(trainer.model, trainer.args, test_loader, trainer.logger)
    
    y_p=np.array(y1[:,0,:].cpu())

    y_t=np.array(y2[:,0,:].cpu())

    y_p = mmn.inverse_transform(y_p)

    y_t = mmn.inverse_transform(y_t)

#y_p=(y_p-mean)/std
#y_t=(y_t-mean)/std

    y_p=torch.tensor(y_p)
    y_t=torch.tensor(y_t)

    diff = y_p[1:] - y_p[:-1]
    return_p = diff / y_p[:-1]

    diff = y_t[1:] - y_t[:-1]
    return_t = diff / y_t[:-1]

    



    
    mae, rmse, _=All_Metrics(return_p,return_t, None,None )

    IC=pearson_correlation(return_t,return_p)
    
    RIC=rank_information_coefficient(return_t[:,0],return_p[:,0])


    result_train_file = os.path.join("AGCRN_Model", "milan","call")



    save_model(trainer,result_train_file,i+1)

    with open('samba_IXIC.txt', 'a') as f:
        f.write(str(np.array(IC)))
        f.write('\n')
        f.write(str(np.array(RIC)))
        f.write('\n')
        f.write(str(np.array(mae)))
        f.write('\n')
        f.write(str(np.array(rmse)))
        f.write('\n\n')