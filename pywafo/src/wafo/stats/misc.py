from numpy import asarray, ndarray, ones, nan #, reshape, repeat,  product

def valarray(shape, value=nan, typecode=None):
    """Return an array of all value.
    """
    #out = reshape(repeat([value],product(shape,axis=0),axis=0),shape)
    out = ones(shape, dtype=bool) * value
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, ndarray):
        out = asarray(out)
    return out

