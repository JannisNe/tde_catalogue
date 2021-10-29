import argparse

from tde_catalogue import main_logger
from tde_catalogue.data.mir_flares.sdss_parnet_sample import SDSSParentSample


logger = main_logger.getChild(__name__)


class SDSSPhotometricGalaxies(SDSSParentSample):
    base_name = SDSSParentSample.base_name + '_photometric_galaxies'
    casjobs_table_name = 'Galaxy'

    default_keymap = {
        'ra': 'ra',
        'dec': 'dec',
        'id': 'ObjID'
    }

    def __init__(self,
                 base_name=base_name,
                 store=True,
                 submit_context='DR16',
                 download_context='DR16',
                 limit_top=100000):
        self.limit_top = limit_top
        super().__init__(base_name=base_name,
                         store=store,
                         submit_context=submit_context,
                         download_context=download_context)

    @property
    def query(self):
        raise NotImplementedError("This should not happen")

    @property
    def _download_query(self):
        _top = f"TOP {self.limit_top}" if self.limit_top else ""
        q = f"""
        SELECT {_top}
            g.ra, g.dec, g.ObjID, g.specObjID, w.cntr as AllWISE_id, xm.match_dist as sep_to_WISE_source
        FROM 
            {self.casjobs_table_name} as g
        JOIN wise_xmatch AS xm ON xm.sdss_objid = g.objid 
        JOIN wise_allsky AS w ON xm.wise_cntr = w.cntr
        """
        return q

    @property
    def _table_in_casjobs(self):
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logging_level', type=str, default='INFO')
    cfg = parser.parse_args()

    main_logger.setLevel(cfg.logging_level)

    SDSSPhotometricGalaxies()