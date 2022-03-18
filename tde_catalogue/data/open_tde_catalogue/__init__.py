import os
import pandas as pd
import numpy as np

from tde_catalogue import main_logger
from tde_catalogue.catalogue import Catalogue


logger = main_logger.getChild(__name__)

raw_data = f'{os.path.dirname(__file__)}/TheOpenTDECatalog.csv'


@Catalogue.register_catalogue('OpenTDECatalogue')
class OpenTDECatalogue(Catalogue):

    def __init__(self):
        df = pd.read_csv(raw_data)

        # exclude TNS sources, that are weird in tde.space
        m = df.Name == "AT"
        m2 = df.Name == "ZTF18abxftqm"
        df = df[~m & (~m2)]

        # exclude sources without date

        super().__init__(df, f'OpenTDECatalogue')

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

        date = list()
        for x in unformated_catalogue[key_map['Discovery Date (UT)']]:
            if isinstance(x, str):
                date.append(x.replace('/', '-'))
            else:
                date.append(x)

        reformatted_catalogue['Discovery Date (UT)'] = date

        for k in Catalogue.columns:
            if k == 'Discovery Date (UT)':
                continue

            if k in key_map:
                reformatted_catalogue[k] = unformated_catalogue[key_map[k]]
            else:
                reformatted_catalogue[k] = [np.nan] * len(unformated_catalogue)

        reformatted_catalogue["Ext. catalog/s"] = "OpenTDECatalogue"

        return reformatted_catalogue