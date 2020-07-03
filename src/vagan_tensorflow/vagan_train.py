# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Modified by Ricardo Bigolin Lanfredi 
# Last modified: 2020-07-01

from . import adni_experiment as exp_config
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

from .model_vagan import vagan
import os

def main():
    # Get Data
    if exp_config.data_identifier == 'adni':
        from .adni_data import adni_data as data_loader
    else:
        raise ValueError('Unknown data identifier: %s' % exp_config.data_identifier)

    data = data_loader(exp_config)

    # Build VAGAN model
    vagan_model = vagan(exp_config=exp_config, data=data, fixed_batch_size=exp_config.batch_size)
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    # Train VAGAN model
    vagan_model.train()


if __name__ == '__main__':

    main()


