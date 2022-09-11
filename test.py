# import os
# os.chdir('ffsb')
# test loading env


try:
    from firefly_task import ffacc_real
    from env_config import Config
    arg = Config()
    env=ffacc_real.FireFlyPaper(arg)
except ImportError:
    raise ImportError('broken task env!!')

# test loading plot functions
try:
    from plot_ult import *
except ImportError:
    raise ImportError('broken plotting env!!')


print('everythings good!')