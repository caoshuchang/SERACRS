import json
import os
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils import padded_tensor
class CRSRecDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,
        use_resp=False, preference_file=None,
    ):
        super(CRSRecDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.use_resp = use_resp
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length
        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length
        dataset_dir = os.path.join('/root/autodl-tmp/MSCRS/rec_data', dataset)
        if preference_file is None:
            preference_file = os.path.join(dataset_dir, 'conversation_preferences.json')
        self.preferences = {}
        self.preference_events = {}
        if os.path.isfile(preference_file):
            with open(preference_file, encoding='utf-8') as stream:
                payload = json.load(stream)
                self.preferences = payload.get('preferences', {})
                self.preference_events = payload.get('events', {})
        self.dataset = dataset
        self.split = split
        preferred = os.path.join(dataset_dir, f'{split}_data.jsonl')
        inspired = os.path.join(dataset_dir, f'{split}_data_train.jsonl')
        data_file = preferred if os.path.isfile(preferred) else inspired
        if not os.path.isfile(data_file):
            raise FileNotFoundError(
                f"no recommendation data for {dataset}/{split}: tried {preferred} and {inspired}"
            )
        self.data = []
        self.prepare_data(data_file)

    @staticmethod
    def _conversation_uid(split, conversation_id):
        # INSPIRED restarts conv_id at zero in every split.  A split namespace
        # prevents validation/test ids from accidentally matching train scenes.
        offsets = {'train': 0, 'valid': 1_000_000_000, 'test': 2_000_000_000}
        return offsets.get(split, 3_000_000_000) + int(conversation_id)

    def _preference(self, conversation_id):
        keys = (f'{self.split}:{conversation_id}', str(conversation_id))
        for key in keys:
            if key in self.preferences:
                return self.preferences[key]
        return {}

    def _visible_preference(self, conversation_id, context_turn_count):
        """Return labels whose supporting user feedback is already in context."""
        preference = self._preference(conversation_id)
        event_key = f'{self.split}:{conversation_id}'
        events = self.preference_events.get(event_key)
        if not events:
            return preference
        visible = {'positive': [], 'negative': [], 'neutral': []}
        for movie_id, event in events.items():
            if int(event['available_from_context_length']) <= int(context_turn_count):
                visible[event['label']].append(int(movie_id))
        return visible

    def _encode_inspired_context(self, turns):
        context = ''
        prompt_context = ''
        for index, utterance in enumerate(turns):
            if not utterance:
                continue
            speaker = 'User: ' if index % 2 == 0 else 'System: '
            context += speaker + utterance + self.tokenizer.eos_token
            prompt_context += speaker + utterance + self.prompt_tokenizer.sep_token
        return context, prompt_context

    def prepare_data(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:min(2000, len(lines))]
            for line in tqdm(lines):
                dialog = json.loads(line)
                if 'context_tokens' in dialog:
                    context = ' '.join(dialog['context_tokens'])
                    prompt_context = context
                    targets = [int(dialog['items'])]
                    entities = dialog.get('context_entities', [])[-self.entity_max_length:]
                    identity = dialog.get('identity', '-1/-1')
                    conversation_id = str(identity).split('/', 1)[0]
                else:
                    if not dialog.get('rec') or not any(dialog.get('context', [])):
                        continue
                    context, prompt_context = self._encode_inspired_context(dialog['context'])
                    targets = [int(item) for item in dialog['rec']]
                    entities = dialog.get('entity', [])[-self.entity_max_length:]
                    conversation_id = str(dialog['conv_id'])
                    identity = f'{conversation_id}/{len(dialog.get("context", []))}'
                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.context_max_length:]
                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)
                count = defaultdict(int)
                for en in dialog.get('retrieved_response_entity', []) + dialog.get('retrieved_context_entity', []):
                    count[en] +=1
                dic = {k:v for k, v in count.items() if v >= 1}
                entitylist = list(dic.keys())
                context_turn_count = len(dialog.get('context', [])) if 'context' in dialog else 10 ** 9
                preference = self._visible_preference(conversation_id, context_turn_count)
                positive_ids = {int(entity) for entity in preference.get('positive', [])}
                negative_ids = {int(entity) for entity in preference.get('negative', [])}
                neutral_ids = {int(entity) for entity in preference.get('neutral', [])}
                preference_labels = []
                for entity in entities:
                    entity = int(entity)
                    if entity in positive_ids:
                        preference_labels.append(0)
                    elif entity in negative_ids:
                        preference_labels.append(1)
                    elif entity in neutral_ids:
                        preference_labels.append(2)
                    else:
                        preference_labels.append(-100)
                for target in targets:
                    data = {
                        # Stable index in the fully expanded split.  This is
                        # used to look up a pre-generated candidate set during
                        # fixed-negative evaluation and is independent of
                        # batching or model implementation.
                        'sample_id': len(self.data),
                        'context': context_ids,
                        'prompt': prompt_ids,
                        'entity': entities,
                        'rec': target,
                        'identity': identity,
                        'conversation_uid': self._conversation_uid(self.split, conversation_id),
                        'context_entities': entities,
                        # Preference labels are supervision only and are assigned
                        # exclusively to entities already visible in this prefix.
                        'positive_mask': [int(entity) in positive_ids for entity in entities],
                        'negative_mask': [int(entity) in negative_ids for entity in entities],
                        'entity_preference_labels': preference_labels,
                    }
                    self.data.append(data)

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSRecDataCollator:
    def __init__(
        self, tokenizer, device, pad_entity_id, use_amp=False, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None
        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length
        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.pad_entity_id = pad_entity_id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length


    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []
        conversation_batch = []
        sample_id_batch = []
        positive_batch = []
        negative_batch = []
        preference_label_batch = []

        for data in data_batch:
            
            input_ids = data['context']
            context_batch['input_ids'].append(input_ids)
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])
            conversation_batch.append(int(data['conversation_uid']))
            sample_id_batch.append(int(data['sample_id']))
            positive_batch.append(data['positive_mask'])
            negative_batch.append(data['negative_mask'])
            preference_label_batch.append(data['entity_preference_labels'])
            prompt_batch['input_ids'].append(data['prompt'])

        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        context_batch['rec_labels'] = label_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, max_length=self.prompt_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        input_batch['entity'] = entity_batch
        input_batch['entity_mask'] = entity_batch.ne(self.pad_entity_id)
        input_batch['positive_entity_mask'] = padded_tensor(
            positive_batch, pad_idx=0, pad_tail=True, device=self.device
        ).bool()
        input_batch['negative_entity_mask'] = padded_tensor(
            negative_batch, pad_idx=0, pad_tail=True, device=self.device
        ).bool()
        input_batch['entity_preference_labels'] = padded_tensor(
            preference_label_batch, pad_idx=-100, pad_tail=True, device=self.device
        ).long()
        input_batch['conversation_ids'] = torch.as_tensor(conversation_batch, device=self.device)
        input_batch['sample_ids'] = torch.as_tensor(sample_id_batch, device=self.device)

        return input_batch
