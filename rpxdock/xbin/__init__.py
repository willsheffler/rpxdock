from cppimport import import_hook
from cppimport.config import turn_off_strict_prototypes
turn_off_strict_prototypes()
from .xbin import *
from .xbin_util import *
from .smear import *

Xbin = Xbin_float
create_Xbin_nside = create_Xbin_nside_float
