!pip install bert-tensorflow==1.0.1
%tensorflow_version 1.x

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import csv
import re
import keras as k
from sklearn import metrics
import codecs
import numpy as np
from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

from google.colab import drive, auth
drive.mount('/content/drive', force_remount=True)

base_path = '/content/drive/My Drive/multilingual_binary/en'

train_path = os.path.join(base_path, 'hateval/train.csv')
dev_path = os.path.join(base_path, 'hateval/dev.csv')

hurtlex_train_path = os.path.join(base_path, 'hateval/train_hurtlex.csv')
hurtlex_dev_path = os.path.join(base_path, 'hateval/dev_hurtlex.csv')

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
#bert_path = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"

max_seq_length = 50

def parse_training(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            corpus.append(row['text'])
            y.append(int(row['label']))
    return corpus, y

def parse_hurtlex(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    features = []
    with open(fp, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            an = row['an']
            asf = row['asf']
            asm = row['asm']
            cds = row['cds']
            ddf = row['ddf']
            ddp = row['ddp']
            dmc = row['dmc']
            isf = row['is']
            om = row['om']
            orf = row['or']
            pa = row['pa']
            pr = row['pr']
            ps = row['ps']
            qas = row['qas']
            rci = row['rci']
            re = row['re']
            svp = row['svp']
            feature = [an,asf,asm,cds,ddf,ddp,dmc,isf,om,orf,pa,pr,ps,qas,rci,re,svp]
            features.append(feature)
    hurtlex_feature = np.asarray(features)
    return hurtlex_feature

dataTrain, labelTrain = parse_training(train_path)
dataDev, labelDev = parse_training(dev_path)    

train_hurtlex_feature = parse_hurtlex(hurtlex_train_path)
dev_hurtlex_feature = parse_hurtlex(hurtlex_dev_path)

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()

# Convert data to InputExample format

# Create datasets (Only take up to max_seq_length words for memory)
train_text = [' '.join(t.split()[0:max_seq_length]) for t in dataTrain]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]

dev_text = [' '.join(t.split()[0:max_seq_length]) for t in dataDev]
dev_text = np.array(dev_text, dtype=object)[:, np.newaxis]


train_examples = convert_text_to_examples(train_text, labelTrain)
dev_examples = convert_text_to_examples(dev_text, labelDev)

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
(dev_input_ids, dev_input_masks, dev_segment_ids, dev_labels) = convert_examples_to_features(tokenizer, dev_examples, max_seq_length=max_seq_length)

class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="first",
        bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        #bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

# Build model
def build_model(max_seq_length, vocab): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    
    concat_in = Input(shape=(17,), dtype='float32')
    
    bert_inputs = [in_id, in_mask, in_segment]
    
    bert_output = BertLayer(n_fine_tune_layers=3, pooling="first")(bert_inputs)
    
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    
    concat = tf.keras.layers.concatenate([dense, concat_in], axis=-1)

    pred = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
    
    model = tf.keras.models.Model(inputs=[bert_inputs,concat_in], outputs=pred)
    myadam = tf.keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=myadam, metrics=['accuracy'])
    model.summary()
    
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

max_len = 50
max_words = 15000
tok = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tok.fit_on_texts(dataTrain)
vocab = len(tok.word_index)

model = build_model(max_seq_length, vocab)

# Instantiate variables
initialize_vars(sess)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = tf.keras.callbacks.ModelCheckpoint('best_model_ami.ckpt', monitor='val_loss',
                                                 save_weights_only=True,
                                                 verbose=1, save_best_only=True)

model.fit(
    [train_input_ids, train_input_masks, train_segment_ids], 
    train_labels,
    validation_data=([dev_input_ids, dev_input_masks, dev_segment_ids], dev_labels),
    verbose=True,
    callbacks=[es, mc],
    epochs=10,
    batch_size=32
)

#for test_dataset in ['hateval_mig', 'hateval_mis', 'hateval', 'abuseval', 'ami_ibereval', 'ami_evalita', 'founta', 'waseem', 'davidson']:
for test_dataset in ['hateval_mis']:
    test_data_path = os.path.join(base_path, '{0}/'.format(test_dataset))
    test_path = os.path.join(test_data_path, 'test.csv')
    #hurtlex_test_path = os.path.join(test_data_path, 'test_hurtlex.csv')

    dataTest, labelTest = parse_training(test_path)
    #test_hurtlex_feature = parse_hurtlex(hurtlex_test_path)

    test_text = [' '.join(t.split()[0:max_seq_length]) for t in dataTest]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]

    test_examples = convert_text_to_examples(test_text, labelTest)

    (test_input_ids, test_input_masks, test_segment_ids, test_labels) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)

    y_prob = model.predict([test_input_ids, test_input_masks,  test_segment_ids])
    y_pred = np.where(y_prob > 0.5, 1, 0)    

    acc = metrics.accuracy_score(labelTest, y_pred) 
    score_pos = metrics.f1_score(labelTest, y_pred, pos_label=1)
    score_neg = metrics.f1_score(labelTest, y_pred, pos_label=0)
    prec = metrics.precision_score(labelTest, y_pred, pos_label=1)
    rec = metrics.recall_score(labelTest, y_pred, pos_label=1)
    prec_neg = metrics.precision_score(labelTest, y_pred, pos_label=0)
    rec_neg = metrics.recall_score(labelTest, y_pred, pos_label=0)

    print(test_dataset)
    print("Accuracy : "+str(acc))
    print("Precision (1) : "+str(prec))
    print("Recall (1) : "+str(rec))
    print("F1-score (1) : "+str(score_pos))
    print("Precision (0) : "+str(prec_neg))
    print("Recall (0) : "+str(rec_neg))
    print("F1-score (0) : "+str(score_neg))

    CorpusFile = './prediction-base-hateval.tsv'
    CorpusFileOpen = codecs.open(CorpusFile, "w", "utf-8")
    for label in y_pred:
      CorpusFileOpen.write(str(label[0])) 
      CorpusFileOpen.write("\n")   
    CorpusFileOpen.close() 
