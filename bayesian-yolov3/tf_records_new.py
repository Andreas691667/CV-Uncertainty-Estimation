import os
import tensorflow as tf
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Define paths
BASE_DIR = os.getcwd() + '/ECP/day'
TFRECORD_DIR = os.getcwd() + '/data/ecp/tfrecords'
os.makedirs(TFRECORD_DIR, exist_ok=True)

# Define shard details
SPLITS = {
    'train': {'num_shards': 20, 'img_dir': 'img/train', 'label_dir': 'labels/train'},
    'val': {'num_shards': 4, 'img_dir': 'img/val', 'label_dir': 'labels/val'},
}

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def create_tf_example(image_path, label_path):
    # Read image
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        encoded_image = f.read()

    # Read JSON labels
    with open(label_path, 'r') as f:
        label_data = json.load(f)

    # Extract relevant features
    width = label_data['imagewidth']
    height = label_data['imageheight']

    ymin, xmin, ymax, xmax, classes_text, classes = [], [], [], [], [], []

    for obj in label_data['children']:
        if obj['identity'] == 'pedestrian':
            ymin.append(obj['y0'] / height)
            xmin.append(obj['x0'] / width)
            ymax.append(obj['y1'] / height)
            xmax.append(obj['x1'] / width)
            classes_text.append(b'pedestrian')
            classes.append(1)

    feature_dict = {
        'image/encoded': bytes_feature(encoded_image),
        'image/format': bytes_feature(b'png'),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_feature(len(classes)),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def get_shard_path(split, shard_idx, num_shards):
    return os.path.join(TFRECORD_DIR, f'ecp-day-{split}-{shard_idx:05d}-of-{num_shards:05d}.tfrecord')

def write_shard(args):
    shard_idx, split, img_paths, label_paths, num_shards = args
    shard_path = get_shard_path(split, shard_idx, num_shards)
    with tf.io.TFRecordWriter(shard_path) as writer:
        for img_path, label_path in zip(img_paths, label_paths):
            example = create_tf_example(img_path, label_path)
            writer.write(example.SerializeToString())

def create_jobs(split):
    split_info = SPLITS[split]
    img_dir = os.path.join(BASE_DIR, split_info['img_dir'])
    label_dir = os.path.join(BASE_DIR, split_info['label_dir'])

    img_paths, label_paths = [], []
    for city in os.listdir(img_dir):
        city_img_dir = os.path.join(img_dir, city)
        city_label_dir = os.path.join(label_dir, city)
        for fname in os.listdir(city_img_dir):
            if fname.endswith('.png'):
                img_path = os.path.join(city_img_dir, fname)
                label_path = os.path.join(city_label_dir, fname.replace('.png', '.json'))
                if os.path.exists(label_path):
                    img_paths.append(img_path)
                    label_paths.append(label_path)

    num_shards = split_info['num_shards']
    shard_size = len(img_paths) // num_shards

    jobs = []
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = (shard_idx + 1) * shard_size if shard_idx < num_shards - 1 else len(img_paths)
        jobs.append((shard_idx, split, img_paths[start_idx:end_idx], label_paths[start_idx:end_idx], num_shards))

    return jobs

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting TFRecord creation...')
    start = time.time()

    all_jobs = []
    for split in ['val']:
        all_jobs.extend(create_jobs(split))

    with ThreadPoolExecutor() as executor:
        executor.map(write_shard, all_jobs)

    elapsed = time.time() - start
    logging.info(f'TFRecord creation completed in {elapsed // 60:.0f} minutes {elapsed % 60:.0f} seconds.')

if __name__ == "__main__":
    main()
