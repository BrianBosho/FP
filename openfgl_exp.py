import openfgl.config as config


from openfgl.flcore.trainer import FGLTrainer

args = config.args

args.root = "your_data_root"


args.dataset = ["Photo"]
args.simulation_mode = "subgraph_fl_label_skew"
args.num_clients = 10
args.num_rounds = 100

args.dirichlet_alpha = 10000
args.skew_alpha = args.dirichlet_alpha
# args.lr = 0.5
# args.optimizer = "SGD"
# args.hid_dim = 64
# args.num_layers = 2

fed_avg = True


if fed_avg:
    args.fl_algorithm = "fedavg"
    args.model = ["gcn"]
else:
    args.fl_algorithm = "fedstar"
    args.model = ["DecoupledGIN"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = ["accuracy"]



trainer = FGLTrainer(args)

trainer.train()