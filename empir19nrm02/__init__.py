###############################################################################
# Initialze empir19nrm02
###############################################################################
# Package info:
__VERSION__ = 'v0.0.0'; """Current version"""
__version__ = __VERSION__
__DATE__ = '20-Sep-2021'; """release date"""

__COPYRIGHT__ = 'Copyright (C) 2021-2021 - 19nmr02'; """copyright info"""

__AUTHOR__ = 'Udo Krüger'; """Package author"""
__EMAIL__ = 'udo.krueger at technoteam.de'; """contact info"""
__URL__ = 'https://github.com/UdoKrueger/empir19nrm02'; """package url"""
__LICENSE__ = 'GPLv3'; """ License """
__DOI__ = ['doi']; """ DOIs """
__CITE__ = 'cite'; """ Citation info """
__all__ = ['__version__','__VERSION__','__AUTHOR__','__EMAIL__', '__URL__','__DATE__',
           '__COPYRIGHT__','__LICENSE__','__DOI__','__CITE__']

from empir19nrm02 import spectral
__all__ += spectral.__all__
__all__ += ['spectral']

from empir19nrm02 import tools
__all__ += tools.__all__
__all__ += ['tools']

from empir19nrm02 import f1prime
__all__ += f1prime.__all__
__all__ += ['f1prime']

from empir19nrm02.spectral.spectral_data import _SPD, _RES
__all__ += ['_SPD', '_RES']

from empir19nrm02 import MC
__all__ += MC.__all__
__all__ += ['MC']