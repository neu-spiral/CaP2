from source.core.engine import MoP
import source.core.run_partition as run_p
from os import environ

dataset='cifar10'
environ["config"] = f"config/{dataset}.yaml"

configs = run_p.main()
configs['model'] = "resnet18"
configs['device'] = "cpu"
configs['load_model'] = "cifar10-resnet18-kernel-npv2_yarkin.pt"
configs['num_partition'] = ".\\config\\resnet18-v2.yaml"
print(configs)
mop = MoP(configs)