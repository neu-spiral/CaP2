from source.core.engine import MoP
from source.core import run_partition 
from os import environ, path
from source.utils import dataset
from source.utils import misc

# TODO: there is some weird import related error going on here where this script is called twice

# select dataset
data='cifar10'
environ["config"] = path.join('config', f'{data}.yaml')

# add to default configs
configs = run_partition.main()
configs['model'] = "resnet18"
configs['device'] = "cpu"
configs['load_model'] = "cifar10-resnet18-kernel-npv2_yarkin.pt" # make sure this makes sense with selected dataset
configs['num_partition'] = path.join('.', 'config', 'resnet18-v2.yaml')

# make model object
print(configs)
mop = MoP(configs)

# prep data for evaluation
mop.train_loader, mop.test_loader = dataset.get_dataset_from_code(mop.configs['data_code'], mop.configs['batch_size'])
criterion = misc.CrossEntropyLossMaybeSmooth(smooth_eps=mop.configs['smooth_eps']).to(mop.configs['device'])

# test accuracy 
print('Testing model \"'+ configs['load_model'] + '\" on ' + configs['data_code'] )
acc = mop.test_model(mop.model, criterion)