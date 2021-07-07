import requests
import pandas as pd

from tde_catalogue import main_logger


logger = main_logger.getChild(__name__)
mirong_url = 'http://staff.ustc.edu.cn/~jnac/data_public/wisevar.txt'


def get_mirong_sample():
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
    logger.info(f'found {len(mirong_sample)} objects in MIRONG Sample')
    return mirong_sample
