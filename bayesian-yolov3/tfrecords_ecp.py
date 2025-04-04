import logging
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# configure logging to also log to file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s, %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    handlers=[
                        logging.FileHandler('tfrecords_chat.log'),
                        logging.StreamHandler()
                    ])
# Set the logging level to DEBUG
logging.getLogger().setLevel(logging.DEBUG)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class ExampleCreator:
    def __init__(self, out_dir, dataset_name, label_to_text=None):
        self._out_dir = out_dir
        self._dataset_name = dataset_name

        # Create a single Session to run all image coding calls.
        self._sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))

        # Initializes function that decodes RGB PNG data.
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._decoded = tf.image.decode_png(self._decode_data, channels=3)

        self._encode_data = tf.placeholder(dtype=tf.uint8)
        self._encoded = tf.image.encode_png(self._encode_data)
        
        self.label_to_text = label_to_text or [
            'ignore',
            'pedestrian',
            'rider',
            'sitting',
            'unusual',
            'group',
        ]

    def get_shard_filename(self, shard, num_shards, split):
        shard_name = '{}-{}-{:05d}-of-{:05d}'.format(self._dataset_name, split, shard, num_shards)
        return os.path.join(self._out_dir, shard_name)

    def decode_png(self, img_data):
        img = self._sess.run(self._decoded, feed_dict={self._decode_data: img_data})
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        return img

    def encode_png(self, img):
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        assert img.shape[1] == 1920
        assert img.shape[0] == 1024
        return self._sess.run(self._encoded, feed_dict={self._encode_data: img})

    def load_img(self, path):
        with tf.gfile.FastGFile(path, 'rb') as f:
            img_data = f.read()
        return self.decode_png(img_data)

    def load_annotations(self, path):
        with open(path, 'r') as f:
            return json.load(f)


    def create_example(self, img_path, label_path):
        # Read image
        img = self.load_img(img_path)
        annotations = self.load_annotations(label_path)
                
        img_height, img_width = img.shape[:2]
        encoded = self.encode_png(img)
        
        ymin, xmin, ymax, xmax, label, text, inst_id = [], [], [], [], [], [], []
        
        skipped_annotations = 0
        box_cnt = 0

        for obj in annotations['children']:
            if obj['identity'] == 'pedestrian' or obj['identity'] == 'sitting' or obj['identity'] == 'unusual':
                class_label = 1
                ymin.append(obj['y0'] / img_height)
                xmin.append(obj['x0'] / img_width)
                ymax.append(obj['y1'] / img_height)
                xmax.append(obj['x1'] / img_width)
                label.append(class_label)
            elif obj['identity'] == 'rider':
                class_label = 2
                ymin.append(obj['y0'] / img_height)
                xmin.append(obj['x0'] / img_width)
                ymax.append(obj['y1'] / img_height)
                xmax.append(obj['x1'] / img_width)
                label.append(class_label)
            else:
                skipped_annotations += 1
                continue
            
            box_cnt += 1
            
            label_text = self.label_to_text[class_label]
            text.append(label_text.encode('utf8'))
            inst_id.append(int(label_path.split('_')[-1].split('.')[0]))

        if skipped_annotations > 0:
            logging.debug('Skipped {}/{} annotations in {}'.format(skipped_annotations, len(annotations['children']), label_path))

        feature_dict = {
            'image/height': int64_feature(img_height),
            'image/width': int64_feature(img_width),
            'image/filename': bytes_feature(img_path.encode('utf8')),
            'image/source_id': bytes_feature(img_path.encode('utf8')),
            'image/encoded': bytes_feature(encoded),
            'image/format': bytes_feature('png'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmin),
            'image/object/bbox/xmax': float_list_feature(xmax),
            'image/object/bbox/ymin': float_list_feature(ymin),
            'image/object/bbox/ymax': float_list_feature(ymax),
            'image/object/class/text': bytes_list_feature(text),
            'image/object/class/label': int64_list_feature(label),
            'image/object/instance/id': int64_list_feature(inst_id),
            'image/object/cnt': int64_feature(box_cnt),
        }
        
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example, skipped_annotations

def write_shard(args):
    shard, num_shards, split, data, example_creator = args
    out_file = example_creator.get_shard_filename(shard, num_shards, split)
    
    writer = tf.python_io.TFRecordWriter(out_file)
    logging.info('Creating shard {}-{}/{}'.format(split, shard, num_shards))
    logging.info('Writing {} examples to {}'.format(len(data), out_file))
    
    cnt = 0
    skipped_annotations = 0
    for img_path, label_path in tqdm(data, desc=f"Processing shard {shard}/{num_shards}"):
        example, skipped_annotations = example_creator.create_example(img_path, label_path)
        skipped_annotations += skipped_annotations
        writer.write(example.SerializeToString())

        if cnt % 100 == 0:
            logging.info('Writing example {}/{}'.format(cnt, len(data)))
        cnt += 1

    if skipped_annotations > 0:
        logging.info('Skipped {} annotations in {}'.format(skipped_annotations, out_file))

    writer.close()
    logging.info('Finished shard {}-{}/{}'.format(split, shard, num_shards))


def create_jobs(split, img_dir, label_dir, num_shards, example_creator):
    img_paths = []
    label_paths = []

    for root, _, files in os.walk(img_dir):
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                label_path = os.path.join(label_dir, os.path.relpath(img_path, img_dir).replace('.png', '.json'))
                if os.path.exists(label_path):
                    img_paths.append(img_path)
                    label_paths.append(label_path)
                    
    logging.info('Found {} images for {}'.format(len(img_paths), split))

    data = list(zip(img_paths, label_paths))
    k, m = divmod(len(data), num_shards)
    shards = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_shards)]

    jobs = [(shard_id + 1, num_shards, split, shard, example_creator) for shard_id, shard in enumerate(shards)]
    return jobs


def process_dataset(out_dir, dataset_name, img_dir, label_dir, train_shards, val_shards):
    out_dir = os.path.expandvars(out_dir)
    img_dir = os.path.expandvars(img_dir)
    label_dir = os.path.expandvars(label_dir)

    os.makedirs(out_dir, exist_ok=True)

    example_creator = ExampleCreator(out_dir, dataset_name)

    train_img_dir = os.path.join(img_dir, 'train')
    train_label_dir = os.path.join(label_dir, 'train')
    val_img_dir = os.path.join(img_dir, 'val')
    val_label_dir = os.path.join(label_dir, 'val')

    train_jobs = create_jobs('train', train_img_dir, train_label_dir, train_shards, example_creator)
    val_jobs = create_jobs('val', val_img_dir, val_label_dir, val_shards, example_creator)

    jobs = train_jobs + val_jobs
    logging.info('Created {} jobs'.format(len(jobs)))

    with ThreadPoolExecutor() as executor:
        result = executor.map(write_shard, jobs, chunksize=1)


def main():
    config = {
        'out_dir': './DATA/tfrecords_20_4',
        'dataset_name': 'ecp',
        'img_dir': './ecp/ECP/day/img',
        'label_dir': './ecp/ECP/day/labels',
        'train_shards': 20,
        'val_shards': 4,
    }

    logging.info('Saving results to {}'.format(config['out_dir']))
    logging.info('----- START -----')
    start = time.time()

    process_dataset(**config)

    end = time.time()
    elapsed = int(end - start)
    logging.info('----- FINISHED in {:02d}:{:02d}:{:02d} -----'.format(elapsed // 3600,
                                                                       (elapsed // 60) % 60,
                                                                       elapsed % 60))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )

    # main()
    
    # verify the tfrecord files
    path = './data/ecp/tfrecords_20_4/ecp-train-00003-of-00020'
    raw_dataset = tf.data.TFRecordDataset(path)
    
    iterator = raw_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()


    conf = tf.ConfigProto()
    conf.device_count['GPU'] = 1
    conf.log_device_placement = False

    with tf.Session(config=conf) as sess:
        for _ in range(5):
            raw_record = sess.run(next_element)
            example = tf.train.Example()
            example.ParseFromString(raw_record)

            img = example
            # print((img.features.feature))
            print('image/height:', img.features.feature['image/height'].int64_list.value[0])
            print('image/width:', img.features.feature['image/width'].int64_list.value[0])
            print('image/filename:', img.features.feature['image/filename'].bytes_list.value[0].decode('utf-8'))
            print('image/source_id:', img.features.feature['image/source_id'].bytes_list.value[0].decode('utf-8'))
            print('image/encoded: [binary data]')
            print('image/format:', img.features.feature['image/format'].bytes_list.value[0].decode('utf-8'))
            print('image/object/bbox/xmin:', img.features.feature['image/object/bbox/xmin'].float_list.value)
            print('image/object/bbox/xmax:', img.features.feature['image/object/bbox/xmax'].float_list.value)
            print('image/object/bbox/ymin:', img.features.feature['image/object/bbox/ymin'].float_list.value)
            print('image/object/bbox/ymax:', img.features.feature['image/object/bbox/ymax'].float_list.value)
            print('image/object/class/text:', [text.decode('utf-8') for text in img.features.feature['image/object/class/text'].bytes_list.value])
            print('image/object/class/label:', img.features.feature['image/object/class/label'].int64_list.value)
            print('image/object/instance/id:', img.features.feature['image/object/instance/id'].int64_list.value)
            
            print('\n')