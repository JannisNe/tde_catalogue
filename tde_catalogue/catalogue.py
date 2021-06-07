import abc


class Catalogue(abc.ABC):

    columns = [
        "ID",
        "name",
        "Reps",
        "Class",
        "RA",
        "DEC",
        "Obj. Type",
        "Redshift",
        "Hostname",
        "Host Redshift",
        "Reporting Groups",
        "Discovery Data Source/s",
        "Classifying Group/s",
        "Disc. Internal Name",
        "Public",
        "Object Spectra",
        "Discovery Mag/Flux",
        "Discovery Filter",
        "Discovery Date (UT)",
        "Sender"
    ]

    def __init__(self, catalogue, name):
        """
        Base Catalogue class
        :param catalogue: pandas.Dataframe, containing the TDE catalogue
        :param name: str, the name of the catalogue
        """

        self.raw_catalogue = catalogue
        self.catalogue = self.reformat_catalogue(catalogue)
        self.name = name

    def __getitem__(self, item):
        return self.catalogue[item]

    @staticmethod
    @abc.abstractmethod
    def reformat_catalogue(unformated_catalogue):
        """Takes the raw catalogue and returns it in the uniform data format"""
        raise NotImplementedError
