from cppimport import import_hook
from cppimport.config import turn_off_strict_prototypes
turn_off_strict_prototypes()
from ._motif import jagged_bin, logsum_bins, marginal_max_score
from .frames import *
from .pairdat import *
from .pairscore import *
from .rpxgen import *
