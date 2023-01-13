import torch
from torch.utils import data

from data_utils.utils import preprocess_sentence, is_japanese_sentence
from utils.instances import Instances
from builders.dataset_builder import META_DATASET

import json
import os
import numpy as np
from PIL import Image, ImageFile
from typing import Dict, List, Any

ImageFile.LOAD_TRUNCATED_IMAGES = True

class BaseDataset(data.Dataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(BaseDataset, self).__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = vocab

        # quesion-answer pairs
        self.annotations = self.load_annotations(json_data)

        # image features
        self.image_features_path = config.FEATURE_PATH.FEATURES

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        raise NotImplementedError

    def load_features(self, image_id: int) -> Dict[str, Any]:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npy")
        features = np.load(feature_file, allow_pickle=True)[()]
        for key, feature in features.items():
            if isinstance(feature, np.ndarray):
                features[key] = torch.tensor(feature)
        
        return features

    def __getitem__(self, idx: int):
        raise NotImplementedError("Please inherit the BaseDataset class and implement the __getitem__ method")

    def __len__(self) -> int:
        return len(self.annotations)

@META_DATASET.register()
class DictionaryDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(DictionaryDataset, self).__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
               if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answer = ann["answer"]
                    if is_japanese_sentence(question):
                        question = list(question)
                        answer = list(answer)
                    else:
                        question = preprocess_sentence(question, self.vocab.tokenizer)
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                    annotation = {
                        "question_id": ann["id"],
                        # "type": None,
                        "question": question,
                        "answer": answer,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations    

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        question = item["question"]
        question_tokens = self.vocab.encode_question(question)
        answer = item["answer"]

        return Instances(
            question_id=item["question_id"],
            # type=item["type"],
            image_id=image_id,
            filename=filename,
            question=question,
            question_tokens=question_tokens,
            answer=answer,
            **features
        )



@META_DATASET.register()
class MultilingualDictionaryDataset(DictionaryDataset):
    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answer = ann["answer"]
                    if is_japanese_sentence(question):
                        question = list(question)
                        answer = list(answer)
                    else:
                        question = preprocess_sentence(question, self.vocab.tokenizer)
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                    # answers = [" ".join(answer)]
                    annotation = {
                        "question_id": ann["id"],
                        # "type": None,
                        "question": question,
                        "answer": answer,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations    


@META_DATASET.register()
class FeatureDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(FeatureDataset, self).__init__(json_path, vocab, config)

    @property
    def questions(self):
        return [ann["question"] for ann in self.annotations]

    @property
    def answers(self):
        return [ann["answer"] for ann in self.annotations]

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answer = ann["answer"]
                    if is_japanese_sentence(question):
                        question = list(question)
                        answer = list(answer)
                    else:
                        question = preprocess_sentence(question, self.vocab.tokenizer)
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                    answers = [" ".join(answer)]
                    annotation = {
                        "question_id": ann["id"],
                        # "type": None,
                        "question": question,
                        "answer": answers,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = self.vocab.encode_question(item["question"])
        answer = self.vocab.encode_answer(item["answer"])

        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        answer = torch.where(answer == self.vocab.eos_idx, self.vocab.padding_idx, answer) # remove eos_token in answer
        
        features = self.load_features(self.annotations[idx]["image_id"])

        return Instances(
            question_tokens=question,
            answer_tokens=answer,
            shifted_right_answer_tokens=shifted_right_answer,
            **features,
        )

    def __len__(self) -> int:
        return len(self.annotations)



@META_DATASET.register()
class MultilingualFeatureDataset(FeatureDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answer = ann["answer"]
                    if is_japanese_sentence(question):
                        question = list(question)
                        answer = list(answer)
                    else:
                        question = preprocess_sentence(question, self.vocab.tokenizer)
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                    annotation = {
                        "question_id": ann["id"],
                        "question": question,
                        "answer": answer,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    annotations.append(annotation)
                    break

        return annotations



@META_DATASET.register()
class RawQuestionMultilingualFeatureDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    @property
    def questions(self):
        return [ann["question"] for ann in self.annotations]

    @property
    def answers(self):
        return [ann["answer"] for ann in self.annotations]

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    answer = ann["answer"]
                    question = ann["question"]
                    if is_japanese_sentence(question):
                        answer = list(answer)
                    else:
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                    annotation = {
                        "question": question,
                        "answer": answer,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    annotations.append(annotation)
                    break
        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        question = item["question"]
        answer = self.vocab.encode_answer(item["answer"])

        shifted_right_answer = torch.zeros_like(answer).fill_(self.vocab.padding_idx)
        shifted_right_answer[:-1] = answer[1:]
        answer = torch.where(answer == self.vocab.eos_idx, self.vocab.padding_idx, answer) # remove eos_token in answer
        
        features = self.load_features(self.annotations[idx]["image_id"])

        return Instances(
            question=question,
            answer_tokens=answer,
            shifted_right_answer_tokens=shifted_right_answer,
            **features,
        )

    def __len__(self) -> int:
        return len(self.annotations)

@META_DATASET.register()
class RawQuestionMultilingualDictionaryDataset(BaseDataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super(RawQuestionMultilingualDictionaryDataset,self).__init__(json_path, vocab, config)

    def load_annotations(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    question = ann["question"]
                    answer = ann["answer"]
                    if is_japanese_sentence(question):
                        answer = list(answer)
                    else:
                        answer = preprocess_sentence(answer, self.vocab.tokenizer)
                    annotation = {
                        "question_id": ann["id"],
                        # "type": ann["QA-type"],
                        "question": question,
                        "answer": answer,
                        "image_id": ann["image_id"],
                        "filename": image["filename"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        image_id = item["image_id"]
        filename = item["filename"]
        features = self.load_features(image_id)
        question = item["question"]
        answer = item["answer"]

        return Instances(
            question_id=item["question_id"],
            # type=item["type"],
            image_id=image_id,
            filename=filename,
            question=question,
            answer=answer,
            **features
        )