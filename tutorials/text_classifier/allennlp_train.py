
"""" ======================================================ARGUMENTS=================================================== """
# indexer = 'elmochar'  # elmochar, singleid
# embedding_type = 'elmo'  # elmo,glove
# encoder_type = 'lstm'  # lstm,bag
# decoder_type = 'linear'
# data = 'protestnews'  # protestnews, sentiment
#
# train_file = '/content/train.json'
# validation_file = '/content/dev.json'
# out_model_name = 'elmo_freeze_india_2'
# vocab_folder_name = 'vocabulary'
# out_dir = '/content/elmo_freeze/'
#
# #train arg
# seed=42
# batch_size=32
# lr=5e-5
# epochs=3
# max_seq_len=128  # necessary to limit memory usage
# max_vocab_size=100000
# validation_metric='+fmacro'
#
# #lstm arg
# hidden_sz=64  # arbitrary
#
# #elmo arg
# options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# FINETUNE_EMBEDDINGS = False

"""" ======================================================TRAIN=================================================== """
label_cols = ["0", "1"]

import sys
sys.path.insert(0, "allennlp")

import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
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

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("--indexer_type", default=None, type=str, required=True,
                    help="elmochar, singleid")
parser.add_argument("--embedding_type", default=None, type=str, required=True,
                    help="elmo,glove")
parser.add_argument("--encoder_type", default=None, type=str, required=True,
                    help="lstm, bag")
parser.add_argument("--decoder_type", default=None, type=str, required=True,
                    help="linear")
parser.add_argument("--data_type", default=None, type=str, required=True,
                    help="protestnews, sentiment")
parser.add_argument("--train_file_path", default=None, type=str, required=True,
                    help="train file path")
parser.add_argument("--validation_file_path", default=None, type=str, required=True,
                    help="validation file path")
parser.add_argument("--vocab_folder_name", default=None, type=str, required=True,
                    help="folder to save vocab")
parser.add_argument("--out_dir_path", default=None, type=str, required=True,
                    help="out dir")

## Paras having default values
parser.add_argument("--seed", default=42, type=int, required=False,
                    help="seed for randomization")
parser.add_argument("--batch_size", default=32, type=int, required=False,
                    help="batch size")
parser.add_argument("--learning_rate", default=5e-5, type=float, required=False,
                    help="learning rate")
parser.add_argument("--epochs", default=3, type=int, required=False,
                    help="num epochs")
parser.add_argument("--max_seq_len", default=128, type=int, required=False,
                    help="max seq len")
parser.add_argument("--max_vocab_size", default=100000, type=int, required=False,
                    help="max vocab size")
parser.add_argument("--validaton_metric", default="+fmacro", type=str, required=False,
                    help="validation metric to watch.")

## Params for specific cases
## Params for lstm
parser.add_argument("--hidden_sz", default=64, type=int, required=False,
                    help="hidden size for lstm")
## Params for elmo
parser.add_argument("--options_file", default=None, type=str, required=False,
                    help="options file for elmo")
parser.add_argument("--weights_file", default=None, type=str, required=False,
                    help="weight file for elmo")
parser.add_argument("--finetine_embeddings", action='store_true',
                        help="Whether to finetune elmo embeddings.")
args = parser.parse_args()

def tokenizer(x: str):
    return [w.text for w in
            SpacyTokenizer(language='en_core_web_sm',
                           pos_tags=False).tokenize(x)[:config.max_seq_len]]

config = Config(
    seed=args.seed,
    batch_size=args.batch_size,
    lr=args.learning_rate,
    epochs=args.epochs,
    hidden_sz=args.hidden_sz,
    max_seq_len=args.max_seq_len,
    max_vocab_size=args.max_vocab_size,
)

USE_GPU = torch.cuda.is_available()

torch.manual_seed(config.seed)

if args.indexer_type == 'singleid':
    token_indexer = SingleIdTokenIndexer()
elif args.indexer_type == 'elmochar':
    token_indexer = ELMoTokenCharactersIndexer()

if args.data_type == 'sentiment':
    reader = SentimentDatasetReader(
        config=config,
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer}
    )
elif args.data_type ==  'protestnews':
    reader = ProtestNewsDatasetReader(
        config=config,
        tokenizer=tokenizer,
        token_indexers={"tokens": token_indexer}
    )

train_ds = reader.read(args.train_file_path)
val_ds = reader.read(args.validation_file_path)

vocab = Vocabulary.from_instances(train_ds, max_vocab_size=config.max_vocab_size)
vocab.save_to_files(directory=args.out_dir_path+args.vocab_folder_name)

iterator = BucketIterator(batch_size=config.batch_size,
                          sorting_keys=[("text", "num_tokens")],
                          )
iterator.index_with(vocab)


if args.embedding_type == 'glove':
    param_dict = {"pretrained_file": "(https://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.300d.txt",
                  "embedding_dim": 300
                  }
    params = Params(params=param_dict)
    token_embedding = Embedding.from_params(vocab=vocab, params=params)
elif args.embedding_type == 'elmo':
    token_embedding = ElmoTokenEmbedder(args.options_file, args.weight_file, requires_grad=args.finetune_embeddings)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

if args.encoder_type == 'bag':
    encoder = BagOfEmbeddingsEncoder(word_embeddings.get_output_dim())
elif args.encoder_type == 'lstm':
    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(word_embeddings.get_output_dim(), config.hidden_sz, bidirectional=True, batch_first=True))


num_classes = vocab.get_vocab_size("labels")
decoder_input_dim = encoder.get_output_dim()

if args.decoder_type=='linear':
    decoder = torch.nn.Linear(decoder_input_dim, num_classes)

model = TextClassifier(word_embeddings, encoder, decoder, vocab)

if USE_GPU:
    model.cuda()
else:
    model

optimizer = optim.Adam(model.parameters(), lr=config.lr)

trainer = Trainer(
    serialization_dir=args.out_dir_path,
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds,
    validation_dataset=val_ds,
    validation_metric=args.validation_metric,
    cuda_device=0 if USE_GPU else -1,
    num_epochs=config.epochs
)

metrics = trainer.train()
