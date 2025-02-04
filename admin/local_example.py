class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = 'snapshots/'
        # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir  # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir 
        self.pre_trained_models_dir = ''
        self.hp = 'data/hpatches-sequences-release/'
        self.eth3d = 'data/ETH3D/'

