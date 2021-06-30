import unittest

from tde_catalogue import main_logger
from tde_catalogue.compilation.compiler import Compiler, CompilerError
from tde_catalogue.catalogue import Catalogue
import tde_catalogue.data


main_logger.setLevel('DEBUG')
logger = main_logger.getChild(__name__)


class TestCompiler(unittest.TestCase):

    def test_a_duplicate(self):
        logger.info('\n\n Testing duplicate Error \n')
        logger.info(f'using {list(Catalogue.registered_catalogues.keys())[0]}')
        cat_class = list(Catalogue.registered_catalogues.values())[0]
        catalogues = [cat_class()] * 2
        with self.assertRaises(CompilerError):
            Compiler(catalogues)

    def test_b_compilation(self):
        logger.info('\n\n Testing compiler \n')
        logger.info(f'using {list(Catalogue.registered_catalogues.keys())}')
        logger.info('Initialising catalogues')
        catalogues = [cat_class() for cat_class in Catalogue.registered_catalogues.values()]
        Compiler(catalogues)
