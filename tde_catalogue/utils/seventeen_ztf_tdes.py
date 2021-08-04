import requests, tarfile, os, logging
import pandas as pd
import numpy as np
from astropy.table import Table

from tde_catalogue import cache_dir, main_logger


logger = main_logger.getChild(__name__)

url = 'https://arxiv.org/e-print/2001.01409'
directory = os.path.join(cache_dir, 'seventeen_ztf_tdes')
data_directory = os.path.join(directory, 'data')
tables_directory = os.path.join(directory, 'tables')

name_map = pd.DataFrame([
    ('AT2018zr', 'ZTF18aabtxvd', 'NedStark', 58180),
    ('AT2018bsi', 'ZTF18aahqkbt', 'JonStark', 58217),
    ('AT2018hco', 'ZTF18abxftqm', 'SansaStark', 58403.4),
    ('AT2018iih', 'ZTF18acaqdaa', 'JorahMormont', 58449.7),
    ('AT2018hyz', 'ZTF18acpdvos', 'GendryBaratheon', 58428.0),
    ('AT2018lni', 'ZTF18actaqdw', 'AryaStark', 58461.9),
    ('AT2018lna', 'ZTF19aabbnzo', 'CerseiLannister', 58508.6),
    ('AT2019cho', 'ZTF19aakiwze', 'PetyrBalish', 58547.5),
    ('AT2019bhf', 'ZTF19aakswrb', 'Varys', 58542.1),
    ('AT2019azh', 'ZTF17aaazdba', 'JaimeLannister', 58558.5),
    ('AT2019dsg', 'ZTF19aapreis', 'BranStark', 58603.1),
    ('AT2019ehz', 'ZTF19aarioci', 'Brienne', 58612.7),
    ('AT2019eve', 'ZTF19aatylnl', 'CatelynStark', 58613.0),
    ('AT2019mha', 'ZTF19abhejal', 'Bronn', 58704.7),
    ('AT2019meg', 'ZTF19abhhjcc', 'MargaeryTyrell', 58696.7),
    ('AT2019lwu', 'ZTF19abidbya', 'RobbStark', 58691.0),
    ('AT2019qiz', 'ZTF19abzrhgq', 'Melisandre', 58763.4)
], columns=['at_name', 'ztf_name', 'nickname', 'peak_mjd'])


def download_data():
    """
    This downloads all data from the paper https://arxiv.org/abs/2001.01409
    and extracts it into the path given in 'directory'
    """
    logger.debug(f'downloading data from {url}')
    with requests.get(url, stream=True) as r:
        with tarfile.open(fileobj=r.raw, mode='r|*') as tar:
            tar.extractall(path=directory)
    logger.debug('download complete')


def extract_tables():
    """
    Extract all table from the downloaded paper.
    Requires the previous execution of download_data()
    """
    _start = r'\begin{tabular}'
    _end = r'\end{tabular}'
    _input = r'\input{tables/'
    main_texfile = os.path.join(directory, 'main.tex')
    with open(main_texfile, 'r') as f:
        tex = f.read()

    i = 0
    j = 0
    while True:
        logger.debug(f'i={i}')
        itex = tex[i:]
        start_ind = itex.find(_start)
        end_ind = itex.find(_end) + len(_end)
        logger.debug(f'from {start_ind} until {end_ind}')

        if start_ind == -1:
            logger.debug('no more tables')
            break

        tab = itex[start_ind:end_ind]

        if _input in tab:
            logger.debug('input command found')
            input_ind = tab.find(_input) + len(_input)
            input_end = input_ind + tab[input_ind:].find('}')
            logger.debug(f'from {input_ind} until {input_end}')

            input_file = os.path.join(tables_directory, tab[input_ind:input_end])
            logger.debug(f'input file is {input_file}')
            _this_input = _input + tab[input_ind:input_end] + '}'
            with open(input_file, 'r') as f:
                rep = f.read()
                logger.debug(f'replacing \n{_this_input}\nwith\n{rep}')
                tab = tab.replace(_this_input, rep)

        tab_fn = os.path.join(tables_directory, f'{j}.tex')

        logger.debug(f'Table:\n{tab}')
        logger.debug(f'writing to {tab_fn}')
        if logger.getEffectiveLevel() == logging.DEBUG:
            input('continue? ')

        with open(tab_fn, 'w') as f:
            f.write(tab)

        i += end_ind
        j += 1

        if end_ind >= (len(tex) - len(_start)):
            logger.debug('reached teh end of file')
            break

    logger.debug(f'found {j} tables')


def apply_replace(x):
    """Replace latex commands in cells"""
    _replace = [r'\textbf{', '}', r'\bf']
    if isinstance(x, str):
        for _r in _replace:
            x = x.replace(_r, '')
        return x
    else:
        return x


def load_tables():
    """
    Load all tables that were extracted from the download paper.
    Requires the previous execution of extract_tables
    :return: list, a list containing the tables as pandas.DataFrame s
    """
    tabs = list()

    for tf in os.listdir(tables_directory):

        try:
            table_number = int(tf.strip('.tex'))
        except ValueError:
            logger.debug(f'table {tf} not an extracted table. skipping')
            continue

        fn = os.path.join(tables_directory, tf)
        logger.debug(f'loading table number {table_number}: {fn}')
        tab = Table.read(fn).to_pandas()
        tab = tab.applymap(apply_replace)
        tabs.append(tab)

    return tabs


def get_lightcurve_from_nickname(tde_nickname):
    """
    load the lightcurve data for one TDE
    :param tde_nickname: str, ZTF nickname of the TDE
    :param bin_day: float, number of days to bin the data in
    :return: lightcurve of TDE, pandas.DataFrame
    """
    logger.debug(f'getting lightcurve for {tde_nickname}')
    file = os.path.join(data_directory, f'{tde_nickname}.dat')

    if not os.path.isdir(data_directory):
        download_data()

    logger.debug(f'loading {file}')
    df = pd.read_csv(file, sep='\t', skiprows=4, skipinitialspace=True)
    df.columns = [c.strip(' ') for c in df.columns]

    peak_date = float(name_map.peak_mjd[name_map.nickname == tde_nickname].iloc[0])
    df['phase_mjd'] = df['#day_since_peak'] + peak_date

    return df


def get_lightcurve_from_atname(at_name):
    """
    :param at_name: str, AT name of the TDE
    :return: lightcurve of TDE, pandas.DataFrame
    """
    ex = at_name in list(name_map.at_name)
    logger.debug(f'{at_name} in map name {ex}')

    if ex:
        nickname = str(name_map.nickname[name_map.at_name == at_name].iloc[0])
        return get_lightcurve_from_nickname(nickname)
    else:
        logger.warning(f'Not in 17 ZTF TDEs!')


def bin_lightcurve(lightcurve, bin_days,
                   luminosity_key='lum_bb', time_key='phase_mjd'):
    """

    :param lightcurve:
    :param bin_days:
    :param luminosity_key:
    :param time_key:
    :return:
    """

    binned_luminosity_key = luminosity_key + '_binned'
    binned_lc = pd.DataFrame(columns=[time_key, binned_luminosity_key, 'upper', 'lower'])

    tstart = lightcurve[time_key].min()
    tend = lightcurve[time_key].max()
    nintervals = int(np.ceil((tend - tstart) / bin_days))

    bounds = np.array([tstart + i * bin_days for i in range(nintervals + 1)])
    intervals = np.array([bounds[:-1], bounds[1:]]).T

    for interval in intervals:
        m = (lightcurve[time_key] >= interval[0]) & (lightcurve[time_key] <= interval[1])
        lum_bb = lightcurve[luminosity_key][m]

        if len(lum_bb) == 0:
            continue
        elif len(lum_bb) == 1:
            med = lum_bb.iloc[0]
            u, l = med, med
            t = lightcurve[time_key][m].iloc[0]
        else:
            med, u, l = np.quantile(lum_bb, [0.5, 0.8, 0.2])
            t = sum(interval) / 2

        app = pd.DataFrame([(t, med, u, l)], columns=binned_lc.columns)
        binned_lc = binned_lc.append(app)

    return binned_lc


def get_binned_lightcurve_from_at_name(at_name, bin_days, **kwargs):
    lc = get_lightcurve_from_atname(at_name)
    binned_lc = bin_lightcurve(lc, bin_days, **kwargs)
    return lc, binned_lc


def get_duration(at_name, threshold=0.5, band='r', return_lc=False):
    """
    Get a duration of the TDE based on the lightcurve.
    The duration is defined as the time the luminosity is above threshold * max(luminosity)
    This refers to a specified band
    :param at_name: str
    :param threshold: float
    :param band: str
    :param return_lc: bool, if True the part of the LC that is used is also returned
    :return: tuple (float, float), start and end time relative to the peak luminosity
    """
    lc = get_lightcurve_from_atname(at_name)
    selected_mask = [b.startswith(band) for b in lc.band]
    lc_selected = lc[selected_mask]
    ordered_inds = np.argsort(lc_selected['#day_since_peak'])
    lc_selected_ordered = lc_selected.iloc[ordered_inds]

    duration_mask = lc_selected_ordered.lum >= max(lc_selected.lum) * threshold
    lc_above_threshold = lc_selected_ordered.phase_mjd[duration_mask]

    tstart = min(lc_above_threshold)
    tend = max(lc_above_threshold)

    if not return_lc:
        return tstart, tend
    else:
        return tstart, tend, lc_selected_ordered