from typing import Dict, Optional
import torch
import torch.nn.functional as F
import torch.optim as optim
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
import numpy as np
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp.data.iterators import BucketIterator
from allennlp.data.fields import TextField, LabelField
from allennlp.data import Instance
from allennlp.common.file_utils import cached_path
from allennlp.training.trainer import Trainer

torch.manual_seed(1)

@DatasetReader.register("sentiment_data")
class SentimentDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def get_label_column_name(self):
        return 'label'

    def get_text_column_name(self):
        return 'text'

    def get_id_column_name(self):
        return 'id'

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field }
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path):
        df = self._read_excel(file_path,
                        self.get_id_column_name(),
                        self.get_label_column_name(),
                        self.get_text_column_name())


        texts = df[self.get_text_column_name()]
        labels = df[self.get_label_column_name()]
        ids = df[self.get_id_column_name()]

        instances = []
        for i,text in enumerate(texts):
            label = str(labels[i]).strip()
            text = str(text).strip()
            id = ids[i]
            if '' in [str(id).strip(), text, label]: continue
            # yield self.text_to_instance(text,label)
            instances.append(self.text_to_instance(text,label))
        return instances

@DatasetReader.register("protest_news")
class ProtestNewsDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def get_label_column_name(self):
        return 'label'

    def get_text_column_name(self):
        return 'text'

    def get_id_column_name(self):
        return 'url'

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field }
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path):
        df = self._read_json(file_path,
                        self.get_id_column_name(),
                        self.get_label_column_name(),
                        self.get_text_column_name())


        texts = df[self.get_text_column_name()]
        labels = df[self.get_label_column_name()]
        ids = df[self.get_id_column_name()]

        instances = []
        for i,text in enumerate(texts):
            label = str(labels[i]).strip()
            text = str(text).strip()
            id = ids[i]
            if '' in [str(id).strip(), text, label]: continue
            # yield self.text_to_instance(text,label)
            instances.append(self.text_to_instance(text, label))
        return instances

@Model.register("my_text_classifier")
class MyTextClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 seq2vec_encoder: Seq2VecEncoder,
                 ) -> None: #initializer: InitializerApplicator = InitializerApplicator(),
                 #regularizer: Optional[RegularizerApplicator] = None


        # super().__init__(vocab, regularizer)
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.seq2vec_encoder = seq2vec_encoder
        self.classifier_input_dim = self.seq2vec_encoder.get_output_dim()
        self.classification_layer = torch.nn.Linear(self.classifier_input_dim, self.num_classes)
        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()
        # initializer(self)

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_title = self.text_field_embedder(text)
        text_mask = get_text_field_mask(text)
        encoded_text = self.title_encoder(embedded_title, text_mask)

        logits = self.classifier(encoded_text, dim=-1)
        class_probabilities = F.softmax(logits)

        output_dict = {"class_probabilities": class_probabilities}

        if label is not None:
            loss = self.loss(logits, label.squeeze(-1))
            for metric in self.metrics.values():
                metric(logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

train_file_path = "/home/mkbb/Desktop/fast_reach/data/protestnews_data/edited_jsons/train.json"
validation_file_path = "/home/mkbb/Desktop/fast_reach/data/protestnews_data/edited_jsons/dev.json"
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo_token_indexer = ELMoTokenCharactersIndexer()
reader = ProtestNewsDatasetReader(token_indexers={'tokens': elmo_token_indexer})
train_dataset = reader.read(train_file_path)
validation_dataset = reader.read(validation_file_path)
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
# vocab = Vocabulary()
token_embedding = ElmoTokenEmbedder(options_file=options_file,weight_file=weight_file,requires_grad=True)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
# "abstract_encoder": {
# "type": "lstm",
# "bidirectional": true,
# "input_size": 100,
# "hidden_size": 100,
# "num_layers": 1,
# "dropout": 0.2
# },
# input_size: The number of expected features in the input `x`
# hidden_size: The number of features in the hidden state `h`
input_size = 1024
hidden_size = 512

lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True))
model = MyTextClassifier(vocab, word_embeddings, lstm)

if torch.cuda.is_available():
    cuda_device = 0
    #### Since we do, we move our model to GPU 0.
    model = model.cuda(cuda_device)
else:
    #### In this case we don't, so we specify -1 to fall back to the CPU. (Where the model already resides.)
    cuda_device = -1

optimizer = optim.Adam(model.parameters())
iterator = BucketIterator(batch_size=32, sorting_keys=[("text", "num_tokens")])
iterator.index_with(vocab)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=40,
                  cuda_device=cuda_device)
trainer.train()