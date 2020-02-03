import os
try:
    import matlab_wrapper
except:
    raise ImportError('matlab_wrapper not available. Plaese install it with: '
                      'pip install matlab_wrapper')


class Matlab(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Matlab, cls).__new__(cls, *args, **kwargs)

        return cls._instance

    def __init__(self, *args, **kwargs):
        root = kwargs.get('root', None)
        if not root:
            root = os.getenv('MATLAB_ROOT')
        print('Starting Matlab Session')
        self.session = matlab_wrapper.MatlabSession(matlab_root=root)
