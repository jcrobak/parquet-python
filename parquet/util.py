import os


def sep_from_open(opener):
    if opener in [default_open, default_openw]:
        return os.sep
    else:
        return '/'


def default_openw(f):
    return open(f, 'wb')


def default_mkdirs(f):
    os.makedirs(f, exist_ok=True)


def default_open(f):
    return open(f, 'rb')
