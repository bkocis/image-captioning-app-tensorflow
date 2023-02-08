import concurrent.futures
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import string
import time
import urllib.request

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds


class TrainerClass:

    def load_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SHAPE[:-1])
        return img

    def standardize(self, s):
        s = tf.strings.lower(s)
        s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
        s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
        return s

    def match_shapes(self, images, captions):
        caption_shape = einops.parse_shape(captions, 'b c')
        captions = einops.rearrange(captions, 'b c -> (b c)')
        images = einops.repeat(
            images, 'b ... -> (b c) ...',
            c=caption_shape['c'])
        return images, captions

    def prepare_txt(self, images, texts):
        tokens = tokenizer(texts)

        input_tokens = tokens[..., :-1]
        label_tokens = tokens[..., 1:]
        return (images, input_tokens), label_tokens

    def prepare_dataset(self, ds, tokenizer, batch_size=32, shuffle_buffer=1000):
        # Load the images and make batches.
        ds = (ds
              .shuffle(10000)
              .map(lambda path, caption: (self.load_image(path), caption))
              .apply(tf.data.experimental.ignore_errors())
              .batch(batch_size))

        def to_tensor(inputs, labels):
            (images, in_tok), out_tok = inputs, labels
            return (images, in_tok.to_tensor()), out_tok.to_tensor()

        return (ds
                .map(self.match_shapes, tf.data.AUTOTUNE)
                .unbatch()
                .shuffle(shuffle_buffer)
                .batch(batch_size)
                .map(self.prepare_txt, tf.data.AUTOTUNE)
                .map(to_tensor, tf.data.AUTOTUNE)
                )

    def flickr8k(self, path='flickr8k'):
        path = pathlib.Path(path)

        if len(list(path.rglob('*'))) < 16197:
            tf.keras.utils.get_file(
                origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
                cache_dir='.',
                cache_subdir=path,
                extract=True)
            tf.keras.utils.get_file(
                origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
                cache_dir='.',
                cache_subdir=path,
                extract=True)

        captions = (path/"Flickr8k.token.txt").read_text().splitlines()
        captions = (line.split('\t') for line in captions)
        captions = ((fname.split('#')[0], caption) for (fname, caption) in captions)

        cap_dict = collections.defaultdict(list)
        for fname, cap in captions:
            cap_dict[fname].append(cap)

        train_files = (path/'Flickr_8k.trainImages.txt').read_text().splitlines()
        train_captions = [(str(path/'Flicker8k_Dataset'/fname), cap_dict[fname]) for fname in train_files]

        test_files = (path/'Flickr_8k.testImages.txt').read_text().splitlines()
        test_captions = [(str(path/'Flicker8k_Dataset'/fname), cap_dict[fname]) for fname in test_files]

        train_ds = tf.data.experimental.from_list(train_captions)
        test_ds = tf.data.experimental.from_list(test_captions)

        return train_ds, test_ds

    def conceptual_captions(*, data_dir="conceptual_captions", num_train, num_val):
        def iter_index(index_path):
            with open(index_path) as f:
                for line in f:
                    caption, url = line.strip().split('\t')
                    yield caption, url

        def download_image_urls(data_dir, urls):
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=100)

            def save_image(url):
                hash = hashlib.sha1(url.encode())
                # Name the files after the hash of the URL.
                file_path = data_dir / f'{hash.hexdigest()}.jpeg'
                if file_path.exists():
                    # Only download each file once.
                    return file_path

                try:
                    result = requests.get(url, timeout=5)
                except Exception:
                    file_path = None
                else:
                    file_path.write_bytes(result.content)
                return file_path

            result = []
            out_paths = ex.map(save_image, urls)
            for file_path in tqdm.tqdm(out_paths, total=len(urls)):
                result.append(file_path)

            return result

        def ds_from_index_file(index_path, data_dir, count):
            data_dir.mkdir(exist_ok=True)
            index = list(itertools.islice(iter_index(index_path), count))
            captions = [caption for caption, url in index]
            urls = [url for caption, url in index]

            paths = download_image_urls(data_dir, urls)

            new_captions = []
            new_paths = []
            for cap, path in zip(captions, paths):
                if path is None:
                    # Download failed, so skip this pair.
                    continue
                new_captions.append(cap)
                new_paths.append(path)

            new_paths = [str(p) for p in new_paths]

            ds = tf.data.Dataset.from_tensor_slices((new_paths, new_captions))
            ds = ds.map(lambda path, cap: (path, cap[tf.newaxis]))  # 1 caption per image
            return ds

        data_dir = pathlib.Path(data_dir)
        train_index_path = tf.keras.utils.get_file(
            origin='https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv',
            cache_subdir=data_dir,
            cache_dir='.')

        val_index_path = tf.keras.utils.get_file(
            origin='https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv',
            cache_subdir=data_dir,
            cache_dir='.')

        train_raw = ds_from_index_file(train_index_path, data_dir=data_dir / 'train', count=num_train)
        test_raw = ds_from_index_file(val_index_path, data_dir=data_dir / 'val', count=num_val)

        return train_raw, test_raw

    def save_dataset(self, ds, save_path, image_model, tokenizer, shards=10, batch_size=32):
        # Load the images and make batches.
        ds = (ds
              .map(lambda path, caption: (self.load_image(path), caption))
              .apply(tf.data.experimental.ignore_errors())
              .batch(batch_size))

        # Run the feature extractor on each batch
        # Don't do this in a .map, because tf.data runs on the CPU.
        def gen():
            for (images, captions) in tqdm.tqdm(ds):
                feature_maps = image_model(images)

                feature_maps, captions = self.match_shapes(feature_maps, captions)
                yield feature_maps, captions

        # Wrap the generator in a new tf.data.Dataset.
        new_ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=image_model.output_shape),
                tf.TensorSpec(shape=(None,), dtype=tf.string)))

        # Apply the tokenization
        new_ds = (new_ds
                  .map(self.prepare_txt, tf.data.AUTOTUNE)
                  .unbatch()
                  .shuffle(1000))

        # Save the dataset into shard files.
        def shard_func(i, item):
            return i % shards

        new_ds.enumerate().save(save_path, shard_func=shard_func)

    def load_dataset(self, save_path, batch_size=32, shuffle=1000, cycle_length=2):
        def custom_reader_func(datasets):
            datasets = datasets.shuffle(1000)
            return datasets.interleave(lambda x: x, cycle_length=cycle_length)

        ds = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)

        def drop_index(i, x):
            return x

        ds = (ds
              .map(drop_index, tf.data.AUTOTUNE)
              .shuffle(shuffle)
              .padded_batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))
        return ds


if __name__ == '__main__':
    trainer = TrainerClass()

    choose = 'flickr8k'

    if choose == 'flickr8k':
        train_raw, test_raw = trainer.flickr8k()
    else:
        train_raw, test_raw = trainer.conceptual_captions(num_train=10000, num_val=5000)

    # feature extractor
    IMAGE_SHAPE = (224, 224, 3)
    mobilenet = tf.keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SHAPE,
        include_top=False,
        include_preprocessing=True)
    mobilenet.trainable = False

    # test_img_batch = trainer.load_image(ex_path)[tf.newaxis, :]

    # tokenizer
    vocabulary_size = 5000
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=trainer.standardize,
        ragged=True)
    tokenizer.adapt(train_raw.map(lambda fp, txt: txt).unbatch().batch(1024))

    # mapper for words to indices and indices to words.
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)

    train_ds = trainer.prepare_dataset(train_raw, tokenizer)
    test_ds = trainer.prepare_dataset(test_raw, tokenizer)

    trainer.save_dataset(train_raw, 'train_cache', mobilenet, tokenizer)
    trainer.save_dataset(test_raw, 'test_cache', mobilenet, tokenizer)

    train_ds = trainer.load_dataset('train_cache')
    test_ds = trainer.load_dataset('test_cache')
