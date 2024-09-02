from cppimport import import_hook
from rpxdock.geom.bcc import *
from rpxdock.geom.miniball import *
from rpxdock.geom.xform_dist import *
from rpxdock.geom import bcc, miniball, sym
from rpxdock.geom.sym import *
from rpxdock.geom.expand_xforms import *

def xform_dist2(*args):
    c, o = xform_dist2_split(*args)
    return c + o
