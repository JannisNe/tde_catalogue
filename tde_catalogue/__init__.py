import logging, os


# Setting up the Logger
main_logger = logging.getLogger(__name__)
logger_format = logging.Formatter('%(levelname)s - %(name)s - %(asctime)s: %(message)s', "%H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logger_format)
main_logger.addHandler(stream_handler)
logger = main_logger.getChild(__name__)

# finding the file that contains the setup function tde_catalogue
BASHFILE = os.getenv('TDE_CATALOGUE_BASHFILE', os.path.expanduser('~/.bashrc'))

# Setting up data directory
DATA_DIR_KEY = 'TDE_CATALOGUE_DATA'
if DATA_DIR_KEY in os.environ:
    data_dir = os.environ['TDE_CATALOGUE_DATA']
else:
    logger.warning('TDE_CATALOGUE_DATA not set! Using home directory.')
    data_dir = os.path.expanduser('~/')

output_dir = os.path.join(data_dir, 'output')
plots_dir = os.path.join(output_dir, 'plots')
cache_dir = os.path.join(data_dir, 'cache')

for d in [data_dir, output_dir, plots_dir, cache_dir]:
    if not os.path.isdir(d):
        os.mkdir(d)