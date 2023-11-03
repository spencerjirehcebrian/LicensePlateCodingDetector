import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

filename = 'test.ipynb'
with open(filename) as ff:
    nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
    
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

nb_out = ep.preprocess(nb_in)