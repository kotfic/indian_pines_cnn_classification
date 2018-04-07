import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
stderr = sys.stderr
with open('/dev/null', 'w') as f:
    sys.stderr = f
    import keras
sys.stderr = stderr
