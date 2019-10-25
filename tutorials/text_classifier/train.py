
"""" ======================================================ARGUMENTS=================================================== """

indexer = 'elmochar'  # elmochar, singleid
embedding_type = 'elmo'  # elmo,glove
encoder_type = 'lstm'  # lstm,bag
decoder_type = 'linear'
data = 'protestnews'  # protestnews, sentiment

train_file = '/home/mkbb/Desktop/Berfu/thesis/new_stuff/fast_reach/data/protestnews_data/edited_jsons/train.json'
validation_file = '/home/mkbb/Desktop/Berfu/thesis/new_stuff/fast_reach/data/protestnews_data/edited_jsons/dev.json'
out_model_name = 'elmo_freeze_india_2'
vocab_folder_name = 'vocabulary'
out_dir = '/home/mkbb/Desktop/allennlp_try'
label_cols = ["0", "1"]

#train arg
seed=42
batch_size=32
lr=5e-5
epochs=3
max_seq_len=128  # necessary to limit memory usage
max_vocab_size=100000
validation_metric='+fmacro'

#lstm arg
hidden_sz=64  # arbitrary

#elmo arg
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
FINETUNE_EMBEDDINGS = False

"""" ======================================================TRAIN=================================================== """

import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.insert(0, "allennlp")

from allennlp.data.tokenizers import SpacyTokenizer
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
from allennlp.common import Params
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, BagOfEmbeddingsEncoder

from allennlp.data.dataset_readers import ProtestNewsDatasetReader, SentimentDatasetReader
from tutorials.text_classifier.config import Config
from allennlp.models import TextClassifier

def tokenizer(x: str):
    return [w.text for w in
            SpacyTokenizer(language='en_core_web_sm',
                           pos_tags=False).tokenize(x)[:config.max_seq_len]]

config = Config(
    seed=seed,
    batch_size=batch_size,
    lr=lr,
    epochs=epochs,
    hidden_sz=hidden_sz,
    max_seq_len=max_seq_len,
    max_vocab_size=max_vocab_size,
)

USE_GPU = torch.cuda.is_available()

torch.manual_seed(config.seed)

if indexer == 'singleid':
    token_indexer = SingleIdTokenIndexer()
elif indexer == 'elmochar':
    token_indexer = ELMoTokenCharactersIndexer()

if data == 'sentiment':
    reader = SentimentDatasetReader(
        config=config,
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer}
    )
elif data ==  'protestnews':
    reader = ProtestNewsDatasetReader(
        config=config,
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer}
    )

train_ds = reader.read(train_file)
val_ds = reader.read(validation_file)

vocab = Vocabulary.from_instances(train_ds, max_vocab_size=config.max_vocab_size)
vocab.save_to_files(directory=out_dir+vocab_folder_name)

iterator = BucketIterator(batch_size=config.batch_size,
                          sorting_keys=[("text", "num_tokens")],
                          )
iterator.index_with(vocab)


if embedding_type == 'glove':
    param_dict = {"pretrained_file": "(https://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.300d.txt",
                  "embedding_dim": 300
                  }
    params = Params(params=param_dict)
    token_embedding = Embedding.from_params(vocab=vocab, params=params)
elif embedding_type == 'elmo':
    token_embedding = ElmoTokenEmbedder(options_file, weight_file, requires_grad=FINETUNE_EMBEDDINGS)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

if encoder_type == 'bag':
    encoder = BagOfEmbeddingsEncoder(word_embeddings.get_output_dim())
elif encoder_type == 'lstm':
    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(word_embeddings.get_output_dim(), config.hidden_sz, bidirectional=True, batch_first=True))

model = TextClassifier(word_embeddings, encoder, vocab)

if USE_GPU:
    model.cuda()
else:
    model

optimizer = optim.Adam(model.parameters(), lr=config.lr)

trainer = Trainer(
    serialization_dir=out_dir + out_model_name,
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds,
    validation_dataset=val_ds,
    validation_metric=validation_metric,
    cuda_device=0 if USE_GPU else -1,
    num_epochs=config.epochs
)

metrics = trainer.train()
