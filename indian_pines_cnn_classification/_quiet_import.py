import sys
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

# Must import this here because tensorflow holds a reference to
# sys.stderr and if we let keras import TF then it will release the
# underlying file handle and throw ugly stacktrace errors. It still
# works, its just UGLY.
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

stderr = sys.stderr
with open('/dev/null', 'w') as f:
    sys.stderr = f
    import keras
sys.stderr = stderr
