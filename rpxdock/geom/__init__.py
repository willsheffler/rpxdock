from cppimport import import_hook

from .bcc import *
from .miniball import *
from .xform_dist import *
from . import bcc, miniball, sym
from .sym import *
from .expand_xforms import *

def xform_dist2(*args):
   c, o = xform_dist2_split(*args)
   return c + o