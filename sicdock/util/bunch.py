__all__ = ("Bunch", "bunchify", "unbunchify")


class Bunch(dict):
    def __contains__(self, k):
        try:
            return dict.__contains__(self, k) or hasattr(self, k)
        except:
            return False

    def __getattr__(self, k):
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self[k]
            except KeyError:
                return None

    def __setattr__(self, k, v):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)

    def __delattr__(self, k):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
            except KeyError:
                raise AttributeError(k)
        else:
            object.__delattr__(self, k)

    def copy(self):
        return Bunch.from_dict(super().copy())

    def toDict(self):
        return unbunchify(self)

    def sub(self, __BUNCH_SUB_ITEMS__=None, **kw):
        if len(kw) is 0 and isinstance(__BUNCH_SUB_ITEMS__, dict):
            kw = __BUNCH_SUB_ITEMS__
        b = self.copy()
        for k, v in kw.items():
            b.__setattr__(k, v)
        return b

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        args = ", ".join(["%s=%r" % (key, self[key]) for key in self.keys()])
        return "%s(%s)" % (self.__class__.__name__, args)

    @staticmethod
    def from_dict(d):
        return bunchify(d)


def bunchify(x):
    if isinstance(x, dict):
        return Bunch((k, bunchify(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(bunchify(v) for v in x)
    else:
        return x


def unbunchify(x):
    if isinstance(x, dict):
        return dict((k, unbunchify(v)) for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        return type(x)(unbunchify(v) for v in x)
    else:
        return x
