import gzip, tqdm, os, logging
import shutil
import numpy as np
import pandas as pd

from timewise import ParentSampleBase
from timewise.general import data_dir, main_logger

logger = main_logger.getChild(__name__)
main_logger.setLevel('DEBUG')


class NEWSParentSample(ParentSampleBase):
    """
    This is an implementation of the NEWS catalogue (Khramtsov et al. A&A, Volume 644, December 2020)
    """

    base_name = 'news_sample'

    data_link = "http://cdsarc.u-strasbg.fr/viz-bin/cat/J/A+A/644/A69#/browse"
    zipped_file = os.path.join(data_dir, 'news.dat.gz')
    file = zipped_file.strip('.gz')
    readme_file = os.path.join(data_dir, 'news_readme.txt')

    download_message = f"The NEWS data has not been found! \n" \
                       f"Please visit \n{data_link} \n" \
                       f"and save the ReadMe file to \n{readme_file} \n" \
                       f"and news.dat.gz to \n{zipped_file}"

    default_keymap = {
        'ra': 'RAdeg',
        'dec': 'DEdeg',
        'id': 'AllWISE_designation'
    }

    def __init__(self):
        super().__init__(base_name=NEWSParentSample.base_name)

        if not os.path.isfile(self.local_sample_copy):
            self.df = self.make_sample()
            self.df.to_csv(self.local_sample_copy)

        else:
            logger.debug(f"loading {self.local_sample_copy}")
            self.df = pd.read_csv(self.local_sample_copy, index_col=0)

    @staticmethod
    def make_sample():
        if not os.path.isfile(NEWSParentSample.file):

            if not os.path.isfile(NEWSParentSample.zipped_file):
                raise FileNotFoundError(NEWSParentSample.download_message)

            logging.debug(f"unzipping {NEWSParentSample.zipped_file}")
            with gzip.open(NEWSParentSample.zipped_file, 'rb') as f_in:
                with open(NEWSParentSample.file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

        logger.debug(f'loading README file {NEWSParentSample.readme_file}')
        with open(NEWSParentSample.readme_file, 'r') as f:
            readme = f.read()
            names = [r[22:22 + 10].replace(' ', '').replace('AllWISE', 'AllWISE_designation') for i, r in
                     enumerate(readme.split('\n')[55:93])
                     if i != 33]
            byte_numbers = [tuple(int(ir) for ir in r[0:8].replace(' ', '').split('-')) for i, r in
                            enumerate(readme.split('\n')[55:93])
                            if i != 33]

            dty = ['<f8'] * 37
            dty[-3] = dty[-2] = dty[-1] = '<U50'

            dtypes = [
                (iname, idty) for idty, iname in zip(dty, names)
            ]

            logger.debug(f'loading NEWS data file {NEWSParentSample.file}')
            with open(NEWSParentSample.file.strip('.gz'), 'r') as f:
                dat = f.read().replace('---', 'NaN').split('\n')[:-1]
                arrdat = np.empty(len(dat), dtype=dtypes)

                for i, r in tqdm.tqdm(enumerate(dat), total=len(dat), desc='converting to floats '):
                    entries = [r[b[0] - 1:b[1]].replace("---", "NaN") for b in byte_numbers]

                    for j, e in enumerate(entries):
                        arrdat[i][names[j]] = e

        df = pd.DataFrame(arrdat)
        df['sep_to_WISE_source'] = 0
        return df