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
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from PIL import Image
import requests
import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds

from model_components import SeqEmbedding, DecoderLayer
from utils import TrainerClass
from train_model import TrainModel


class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
        super().__init__()

        self.dense = tf.keras.layers.Dense(
            units=tokenizer.vocabulary_size(), **kwargs)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens

        self.bias = None

    def adapt(self, ds):
        counts = collections.Counter()
        vocab_dict = {name: id
                      for id, name in enumerate(self.tokenizer.get_vocabulary())}

        for tokens in tqdm.tqdm(ds):
            counts.update(tokens.numpy().flatten())

        counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
        counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())

        counts_arr = counts_arr[:]
        for token in self.banned_tokens:
            counts_arr[vocab_dict[token]] = 0

        total = counts_arr.sum()
        p = counts_arr / total
        p[counts_arr == 0] = 1.0
        log_p = np.log(p)  # log(1) == 0

        entropy = -(log_p * p).sum()

        print()
        print(f"Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}")
        print(f"Marginal entropy: {entropy:0.2f}")

        self.bias = log_p
        self.bias[counts_arr == 0] = -1e9

    def call(self, x):
        x = self.dense(x)
        # An Add layer doesn't work because of the different shapes.
        # This clears the mask, that's okay because it prevents keras from rescaling
        # the losses.
        return x + self.bias


class Captioner(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1,
                 units=256, max_length=50, num_heads=1, dropout_rate=0.1):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary())
        self.index_to_word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True)

        self.seq_embedding = SeqEmbedding(
            vocab_size=tokenizer.vocabulary_size(),
            depth=units,
            max_length=max_length)

        self.decoder_layers = [
            DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
            for n in range(num_layers)]

        self.output_layer = output_layer

    def call(self, inputs):
        image, txt = inputs
        if image.shape[-1] == 3:
            # Apply the feature-extractor, if you get an RGB image.
            image = self.feature_extractor(image)
        # Flatten the feature map
        image = einops.rearrange(image, 'b h w c -> b (h w) c')
        if txt.dtype == tf.string:
            # Apply the tokenizer if you get string inputs.
            txt = tokenizer(txt)
        txt = self.seq_embedding(txt)
        # Look at the image
        for dec_layer in self.decoder_layers:
            txt = dec_layer(inputs=(image, txt))
        txt = self.output_layer(txt)
        return txt

    def simple_gen(self, image, temperature=1):
        initial = self.word_to_index([['[START]']])  # (batch, sequence)
        img_features = self.feature_extractor(image[tf.newaxis, ...])

        tokens = initial  # (batch, sequence)
        for n in range(50):
            preds = self((img_features, tokens)).numpy()  # (batch, sequence, vocab)
            preds = preds[:, -1, :]  # (batch, vocab)
            if temperature == 0:
                next = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
            else:
                next = tf.random.categorical(preds / temperature, num_samples=1)  # (batch, 1)
            tokens = tf.concat([tokens, next], axis=1)  # (batch, sequence)

            if next[0] == self.word_to_index('[END]'):
                break
        words = index_to_word(tokens[0, 1:-1])
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result.numpy().decode()


if __name__ == '__main__':
    trainer = TrainerClass()
    # tokenizer
    tokenizer = trainer.tokenizer()


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

    #trainer.save_dataset(train_raw, 'train_cache', mobilenet, tokenizer)
    #trainer.save_dataset(test_raw, 'test_cache', mobilenet, tokenizer)

    train_ds = trainer.load_dataset('train_cache')
    test_ds = trainer.load_dataset('test_cache')


    # output
    output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', '[START]'))
    # This might run a little faster if the dataset didn't also have to load the image data.
    output_layer.adapt(train_ds.map(lambda inputs, labels: labels))

    # build model
    model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                      units=256, dropout_rate=0.5, num_layers=2, num_heads=2)

    # # generate captions
    # image_url = 'https://tensorflow.org/images/surf.jpg'
    # image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
    # image = trainer.load_image(image_path)
    #
    # for t in (0.0, 0.5, 1.0):
    #   result = model.simple_gen(image, temperature=t)
    #   print(result)

    train_model = TrainModel(model=model, train_ds=train_ds, test_ds=test_ds)
    train_model.model_train()
    print('done')
