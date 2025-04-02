import tensorflow as tf
from tensorflow.python.client import device_lib
print([device.name for device in device_lib.list_local_devices() if device.device_type == 'GPU'])