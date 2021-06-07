import os
import pandas as pd
import numpy as np

from tde_catalogue import main_logger
from tde_catalogue.catalogue import Catalogue


logger = main_logger.getChild(__name__)

raw_data = f'{os.path.dirname(__file__)}/TheOpenTDECatalog.csv'


class OpenTDECatalogue(Catalogue):

    def __init__(self):
        super().__init__(pd.read_csv(raw_data), f'OpenTDECatalogue')

    @staticmethod
    def reformat_catalogue(unformated_catalogue):
        key_map = {
            # "RA": None,
            # "DEC": None,
            "Redshift": "z",
            "Hostname": "Host Name",
            "Disc. Internal Name": "Name",
            "Discovery Date (UT)": "Disc. Date",
            "Obj. Type": "Type"
        }
        logger.warning('No field RA and DEC!')
        reformatted_catalogue = pd.DataFrame(columns=Catalogue.columns)

        for k in Catalogue.columns:
            if k in key_map:
                reformatted_catalogue[k] = unformated_catalogue[key_map[k]]
            else:
                reformatted_catalogue[k] = [np.nan] * len(unformated_catalogue)

        return reformatted_catalogue