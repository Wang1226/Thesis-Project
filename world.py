from parse import parse_args
args = parse_args()


config={}
config['bpr_batch']=args.bpr_batch
config['recdim']=args.recdim
config['layer']=args.layer
config['lr']=args.lr
config['decay']=args.decay
config['dropout']=args.dropout
config['epochs']=args.epochs
config['seed']=args.seed
config['topks']=eval(args.topks)
config['multicore']=args.multicore
config['comment']=args.comment
config['tensorboard']=args.tensorboard
config['load']=args.load
config['loss_mode']=args.loss_mode
config['margin_value']=args.margin_value
config['beta']=args.beta
config['lambda_1']=args.lambda_1


# config['path']='./checkpoints'
# config['model']='lgn'
# config['pretrain']=0
# config['a_fold']=100
# config['testbatch']=100
# config['keepprob']=0.6
# config['dataset']='gowalla'
# config['path']='./checkpoints'

