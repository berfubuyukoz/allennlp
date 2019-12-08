from typing import Dict, List
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, LabelField, MetadataField
from typing import *
from tutorials.text_classifier.config import Config

class ProtestNewsDatasetReader(DatasetReader):
    def __init__(self,
                 config: Config,
                 tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.config = config
        self.max_seq_len = self.config.max_seq_len


    def get_label_column_name(self):
        return 'label'

    def get_text_column_name(self):
        return 'text'

    def get_id_column_name(self):
        return 'url'
    
    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokenized_text = [Token(x) for x in self.tokenizer(text)]
        text_field = TextField(tokenized_text, self.token_indexers)
        fields = {'text': text_field }
        if label is not None:
            fields['label'] = LabelField(label)
        # fields['id'] = MetadataField(text_id)
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