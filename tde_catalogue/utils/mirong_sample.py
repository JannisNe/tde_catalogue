import requests, os, tqdm, io
import pandas as pd
import astropy.units as u
import numpy as np

from tde_catalogue import main_logger, cache_dir
from tde_catalogue.utils.panstarrs_utils import crossmatch_to_panstarrs


logger = main_logger.getChild(__name__)
mirong_url = 'http://staff.ustc.edu.cn/~jnac/data_public/wisevar.txt'
local_copy = os.path.join(cache_dir, 'mirong_sample.csv')


def get_mirong_sample():

    if not os.path.isfile(local_copy):

        logger.info(f'getting MIRONG sample from {mirong_url}')
        r = requests.get(mirong_url)
        lll = list()
        for l in r.text.split('\n')[1:]:
            illl = list()
            for ll in l.split(' '):
                if ll and '#' not in ll:
                    illl.append(ll)
            lll.append(illl)

        mirong_sample = pd.DataFrame(lll[1:-1], columns=lll[0])
        mirong_sample['ra'] = mirong_sample['RA']
        mirong_sample['dec'] = mirong_sample['DEC']
        logger.debug(f'saving to {local_copy}')

        mirong_sample.to_csv(local_copy, index=False)
        logger.info(f'found {len(mirong_sample)} objects in MIRONG Sample')

        logger.debug('matching to Pan-STARRS')

        try:
            match_table = crossmatch_to_panstarrs(local_copy, 3 * u.arcsec)

            mirong_sample['panstarrs_objName'] = [''] * len(mirong_sample)
            for idd in tqdm.tqdm(mirong_sample.ID.unique(), desc='inserting into mirong table'):
                m = match_table._searchID_ - 1 == int(idd)
                nobj = len(match_table[m])

                if nobj != 1:

                    raise Exception(f'Found {nobj} matches for {idd} (type({type(idd)}) '
                                    f'in {np.sort(match_table._searchID_ - 1)}!')

                ind = np.where(m)[0][0]
                mirong_sample.loc[ind, 'panstarrs_objName'] = match_table['objName'].iloc[ind].replace(' ', '')
            logger.debug(f'saving to {local_copy}')

        finally:
            logger.debug(f"removing {local_copy}")
            # os.remove(local_copy)

        mirong_sample.to_csv(local_copy, index=True)

    else:
        logger.debug(f'loading {local_copy}')
        mirong_sample = pd.read_csv(local_copy)

    return mirong_sample
