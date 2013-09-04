from __future__ import print_function
from __future__ import division

import contextlib
from cStringIO import StringIO
import os


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, arg=None):
        super(AttrDict, self).__init__()
        if isinstance(arg, dict):
            self.init_from_dict(arg)

    def init_from_dict(self, d):
        for k, v in d.iteritems():
            if isinstance(v, dict):
                self[k] = AttrDict(v)
            else:
                self[k] = v


@contextlib.contextmanager
def capture_output():
    import sys
    old_out, old_err = sys.stdout, sys.stderr
    try:
        out = [StringIO(), StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


def notify(text=None, title='Model run is done'):
    try:
        import gntp.notifier
    except ImportError:
        raise UserWarning('Could not import gntp.notifier, '
                          'cannot send notification.')
    image_path = os.path.join(os.path.dirname(__file__), 'notify.png')
    image = open(image_path, 'rb').read()
    gntp.notifier.mini(text, title=title, applicationIcon=image)
