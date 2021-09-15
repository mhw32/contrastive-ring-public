import os
import random, torch, numpy
from src.systems import systems
from src.utils.setup import process_config
from src.utils.callbacks import MoCoLRScheduler
import pytorch_lightning as pl

torch.backends.cudnn.benchmark = True

SYSTEM = {
    'PretrainNCESystem': systems.PretrainNCESystem,
    'PretrainMoCoSystem': systems.PretrainMoCoSystem,
    'TransferSystem': systems.TransferSystem,
}


def run(config_path, dataset, gpu_device=None):
    if gpu_device == 'cpu' or not gpu_device:
        gpu_device = None
    
    config = process_config(
        config_path,
        exp_name_suffix=dataset,
    )
    config.data_params.dataset = dataset

    if gpu_device: 
        config.gpu_device = int(gpu_device)

    seed_everything(config.seed, use_cuda=config.cuda)
    SystemClass = SYSTEM[config.system]
    system = SystemClass(config)

    if config.optim_params.scheduler:  # moco scheduler
        lr_callback = globals()[config.optim_params.scheduler](
            initial_lr=config.optim_params.learning_rate,
            max_epochs=config.num_epochs,
            schedule=(
                int(0.68*config.num_epochs),
                int(0.83*config.num_epochs),
            ),
        )
        callbacks = [lr_callback]
    else:
        callbacks = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        period=1,
    )

    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=gpu_device,
        distributed_backend=config.distributed_backend or 'dp',
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        # learning rate callbacks
        callbacks=callbacks,
    )
    trainer.fit(system)


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--gpu-device', default=None)
    args = parser.parse_args()

    gpu_device = str(args.gpu_device) if args.gpu_device else None
    run(args.config, dataset=args.dataset, gpu_device=args.gpu_device)
