import tensorflow as tf


###### FROM DATASET UTILS ######

def decode_img(encoded, shape):
    # decode image and scale to [0, 1)
    img = tf.image.decode_png(encoded, dtype=tf.uint8)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # convert to [0, 1)
    img.set_shape(shape) # shape is HWC        
    return img


def make_parse_fn(config):
    def parse_example(example):
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
            # 'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            # 'image/object/cnt': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        }

        features = tf.parse_single_example(example, features=feature_map)

        img = decode_img(features['image/encoded'], config['full_img_size'])

        # assemble bbox
        xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'], default_value=0)
        ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'], default_value=0)
        xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'], default_value=0)
        ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'], default_value=0)
        bbox = tf.stack([ymin, xmin, ymax, xmax], axis=1)  # we use the standard tf bbox format

        label = tf.cast(tf.sparse_tensor_to_dense(features['image/object/class/label'], default_value=-1),
                        dtype=tf.int32)

        # Note regarding implicit background class:
        # The tensorflow object detection API enforces that the class labels start with 1.
        # The class 0 is reserved for an (implicit) background class.

        # yolo does not need a implicit background class.
        # To ensure compatibility with tf object detection API we support both: class ids starting at 1 or 0.
        implicit_background_class = config['implicit_background_class']
        if implicit_background_class:
            label = label - 1  # shift class 1 -> 0, 2 -> 1, etc...

        return img, bbox, label  # this is a mess

    return parse_example

###### END OF DATASET UTILS ######

# === CONFIG EXAMPLE ===
config = {
    'full_img_size': [256, 256, 3],  # replace with your actual shape
    'implicit_background_class': False,
    'cpu_thread_cnt': 4,
    'batch_size': 1,
    'train': {
        'file_pattern': 'DATA/tfrecords_20_4/ecp-train-*-of-*',  # replace with your actual file pattern
        'num_shards': 10,
        'cache': False,
        'shuffle_buffer_size': 100
    }
}

# === DATASET ===
def create_simple_dataset():
    dataset = tf.data.Dataset.list_files(config['train']['file_pattern'])
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=2)
    dataset = dataset.map(make_parse_fn(config), num_parallel_calls=config['cpu_thread_cnt'])
    dataset = dataset.batch(config['batch_size'])
    dataset = dataset.prefetch(1)
    return dataset

dataset = create_simple_dataset()
iterator = dataset.make_one_shot_iterator()
next_img, next_bbox, next_label = iterator.get_next()

# === CONVOLUTION LAYER ===
def simple_conv(img):
    with tf.device('/GPU:0'):
        conv_weights = tf.Variable(tf.random.truncated_normal([3, 3, 3, 8], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[8]))
        conv = tf.layers.conv2d(img, filters=8, kernel_size=[3, 3], strides=(1, 1), padding='SAME', activation=None)
        return tf.nn.relu(conv + bias)

conv_output = simple_conv(next_img)

# === RUN ===
conf = tf.ConfigProto()
conf.device_count['GPU'] = 1
# conf.gpu_options.allow_growth = True
# conf.allow_soft_placement = True
conf.log_device_placement = True
with tf.Session(config=conf) as sess:
    sess.run(tf.global_variables_initializer())
    try:
        img_val, conv_val = sess.run([next_img, conv_output])
        print("Input shape:", img_val.shape)
        print("Conv output shape:", conv_val.shape)
    except tf.errors.OutOfRangeError:
        print("Done.")




# Keep only one GPU active
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# # GPU memory growth
# conf = tf.ConfigProto()
# conf.device_count['GPU'] = 1
# # conf.gpu_options.allow_growth = True
# # conf.allow_soft_placement = True
# conf.log_device_placement = True

# # Build graph
# input_tensor = tf.placeholder(tf.float32, shape=[None, 8, 8, 1], name='input')
# filter_weights = tf.Variable(tf.random_normal([3, 3, 1, 2]), name='filter')
# conv = tf.layers.conv2d(inputs=input_tensor, filters=2, kernel_size=[3, 3], strides=(1, 1), padding='same', use_bias=False)

# # Input data
# input_data = np.random.rand(1, 8, 8, 1).astype(np.float32)


# Run session
# with tf.Session(config=conf) as sess:
#     sess.run(tf.global_variables_initializer())
#     print("Running convolution on the loaded image...")
#     result = sess.run(conv, feed_dict={input_tensor: image_input})
#     print("Done. Result shape:", result.shape)



# if config['crop']:
#     cropper = ImageCropper(config)
#     config['train']['crop_fn'] = cropper.random_crop_and_sometimes_rescale
#     config['val']['crop_fn'] = cropper.random_crop_and_sometimes_rescale
# model_factory = yolov3_aleatoric(config)
# dataset = TrainValDataset(model_blueprint=model_factory.blueprint, config=config)
# img, gt1, gt2, gt3 = dataset.iterator.get_next()

# Run session
# with tf.Session(config=conf) as sess:
#     sess.run(tf.global_variables_initializer())
#     print("Running convolution on GPU...")
#     result = sess.run(conv, feed_dict={input_tensor: input_data})
#     print("Done. Result shape:", result.shape)
