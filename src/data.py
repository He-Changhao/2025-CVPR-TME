from torch.utils.data import Dataset
from pathlib import Path
import json
import random
import PIL
import torch
import logging

FashioniqPath = './data/FashionIQ/'
CIRRPath = './data/CIRR/'

dataset_registry={}
def registry_dataset(name):
    def wapper(cls):
        dataset_registry[name]=cls
        return cls
    return wapper

def get_dataset(name, *args, **kwargs):
    cls = dataset_registry[name]
    return cls(*args, **kwargs)

@registry_dataset('CIRR')
class CIRR(Dataset):
    def __init__(self, preprocess, split, mode='relative', noise_ratio=0.0):
        """
        CIRR dataset
        Args:
            preprocess (callable): the preprocess function for the image
            split (str): the split of the dataset, should be 'train', 'val', 'test1'
            mode (str, optional): the mode of the dataset, should be 'relative' or 'gallery'. Defaults to 'relative'. The 'gallery' mode is for features precomputing and the 'relative' mode is for training and evaluation.
            noise_ratio (float, optional): the noise ratio for the training dataset. Defaults to 0.0.
        """
        self.preprocess = preprocess
        self.noise_ratio = noise_ratio
        self.split = split
        self.mode = mode

        self.path = Path(CIRRPath)
        with open(self.path / 'captions' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.current_triplets = json.load(f)
        
        with open(self.path / 'captions' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)
        self.current_image_names = list(self.name_to_relpath.keys())
        if self.split == 'train':
            self.shuffle()
            
    def shuffle(self):
        logging.info(f'shuffle data with noise ratio {self.noise_ratio}.')
        num_samples = len(self.current_triplets)
        shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))
        par_p1 = int(len(shuffle_indices) * (1/3))
        par_p2 = int(len(shuffle_indices) * (2/3))
        shuffle_candidate_indices = shuffle_indices[:par_p1]
        shuffle_captions_indices = shuffle_indices[par_p1:par_p2]
        shuffle_target_indices = shuffle_indices[par_p2:]
        noise_candidate = [self.current_triplets[i]['reference'] for i in shuffle_candidate_indices]
        noise_captions = [self.current_triplets[i]['caption'] for i in shuffle_captions_indices]
        noise_target = [self.current_triplets[i]['target_hard'] for i in shuffle_target_indices]
        random.shuffle(noise_candidate)
        random.shuffle(noise_captions)
        random.shuffle(noise_target)
        for idx, i in enumerate(shuffle_candidate_indices):
            self.current_triplets[i]['reference'] = noise_candidate[idx]
        for idx, i in enumerate(shuffle_captions_indices):
            self.current_triplets[i]['caption'] = noise_captions[idx]
        for idx, i in enumerate(shuffle_target_indices):
            self.current_triplets[i]['target_hard'] = noise_target[idx]
        logging.info('done')
        
    def __getitem__(self, index):
        # If input slice, then pack the data as list
        if isinstance(index, slice):
            length = self.__len__()
            start = (0 if index.start is None 
                     else index.start if index.start >= 0 else length + index.start)
            stop = (length if index.stop is None 
                    else index.stop if index.stop >= 0 else length + index.stop)
            start = max(0, min(start, length))
            stop = max(0, min(stop, length))
            step = index.step if index.step else 1
            res = [self.__getitem__(i) for i in range(start, stop, step)]
            return list(zip(*res))

        if self.mode == 'relative':
            group_members = self.current_triplets[index]['img_set']['members']
            reference_name = self.current_triplets[index]['reference']
            rel_caption = self.current_triplets[index]['caption']
            if self.split == 'train':
                target_hard_name = self.current_triplets[index]['target_hard']
                return reference_name, target_hard_name, rel_caption, index
            if self.split == 'val':
                target_hard_name = self.current_triplets[index]['target_hard']
                return reference_name, target_hard_name, rel_caption, group_members
            if self.split == 'test1':
                pair_id = self.current_triplets[index]['pairid']
                return pair_id, reference_name, rel_caption, group_members
            raise ValueError("Split shoue be 'train', 'val' or 'test1'")
        elif self.mode == 'gallery':
            image_name = self.current_image_names[index]
            image_path = self.path / self.name_to_relpath[image_name]
            im = PIL.Image.open(image_path)
            image = self.preprocess(im)
            return image_name, image
        else:
            raise ValueError('Mode should be relative or gallery')
        
    def __len__(self):
        if self.mode == 'relative':
            return len(self.current_triplets)
        if self.mode == 'gallery':
            return len(self.current_image_names)
        raise ValueError("Mode should be relative or gallery")
    
    def load_image_features(self):
        logging.info('load the precomputed features')
        self.images = torch.load(f'./features/CIRR/{self.split}_images.pt')
        with open(f'./features/CIRR/{self.split}_name_to_idx.json', 'r') as f:
            self.name_to_idx = json.load(f)
            
    def load_image_features_tensor_and_dict(self, features, dict):
        self.images = features
        self.name_to_idx = dict
    
    def get_image_features(self, name_list):
        indices = [self.name_to_idx[name] for name in name_list]
        images_features = self.images[indices]
        return images_features
    
@registry_dataset('FashionIQ')
class FashionIQ(Dataset):
    def __init__(self, preprocess: callable, split: str, mode: str='relative', noise_ratio=0.0, specific_dress_type=None):
        """
        FashionIQ dataset
        Args:
            preprocess (callable): the preprocess function for the image.
            split (str): the split of the dataset, should be ['train', 'val', 'test1'].
            mode (str, optional): the mode of the dataset, should be ['relative', 'gallery']. Defaults to 'relative'. The 'gallery' mode is for features precomputing and the 'relative' mode is for training and evaluation.
            noise_ratio (float, optional): the noise ratio for the training dataset. Defaults to 0.0.
            specific_dress_type (str, optional): the specific dress type for the dataset. Defaults to None.
            For FashionIQ, the specific_dress_type is used for the evaluation of the specific dress type.
            When specific_dress_type is None, it will return the images with dress types ['dress', 'shirt', 'toptee']. specific_dress_type should be None when split is 'train'.
        """
        if split == 'train' and specific_dress_type is not None:
            raise ValueError("specific_dress_type should be None when split is 'train'")
        self.mode = mode
        self.split = split
        self.preprocess = preprocess
        self.path = Path(FashioniqPath)
        self.noise_ratio = noise_ratio
        
        self.dress_types = ['dress', 'shirt', 'toptee']
        self.type_to_triplets = {}
        self.triplets = []
        for dress_type in self.dress_types:
            with open(self.path / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.type_to_triplets[dress_type] = json.load(f)
                self.triplets.extend(self.type_to_triplets[dress_type])
        self.image_names: list = []
        self.image_type_names = {}
        if self.split == 'train':
            # shuffle the data across dress types
            self.shuffle()
        # get the image names
            images_names = []
            ref_names = [item['candidate'] for item in self.triplets]
            tag_names = [item['target'] for item in self.triplets]
            images_names.extend(ref_names)
            images_names.extend(tag_names)
            self.image_names = list(dict.fromkeys(images_names)) # Keep the image order and remove replicated image
            logging.info(f"FashionIQ {split} dataset in {mode} mode initialized")
        else:
            for dress_type in self.dress_types:
                with open(self.path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                    self.image_type_names[dress_type] = json.load(f)
                    self.image_names.extend(self.image_type_names[dress_type])
            logging.info(f"FashionIQ {split} dataset in {mode} mode initialized")
        self.set_specific_dress_type(specific_dress_type)
                    
    def set_specific_dress_type(self, dress_type):
        """
        Set the specific dress type for the dataset.
        """
        if self.split == 'train' and dress_type is not None:
            raise ValueError("specific_dress_type should be None when split is 'train'")
        self.current_triplets = self.type_to_triplets[dress_type] if dress_type is not None else self.triplets
        self.current_image_names = self.image_type_names[dress_type] if dress_type is not None else self.image_names
         
    def shuffle(self):
        logging.info(f'shuffle data with noise_ratio {self.noise_ratio}.')
        num_samples = len(self.triplets)
        shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))
        par_p1 = int(len(shuffle_indices) * (1/3))
        par_p2 = int(len(shuffle_indices) * (2/3))
        shuffle_candidate_indices = shuffle_indices[:par_p1]
        shuffle_captions_indices = shuffle_indices[par_p1:par_p2]
        shuffle_target_indices = shuffle_indices[par_p2:]
        noise_candidate = [self.triplets[i]['candidate'] for i in shuffle_candidate_indices]
        noise_captions = [self.triplets[i]['captions'] for i in shuffle_captions_indices]
        noise_target = [self.triplets[i]['target'] for i in shuffle_target_indices]
        random.shuffle(noise_candidate)
        random.shuffle(noise_captions)
        random.shuffle(noise_target)
        for i in shuffle_candidate_indices:
            self.triplets[i]['candidate'] = noise_candidate.pop()
        for i in shuffle_captions_indices:
            self.triplets[i]['captions'] = noise_captions.pop()
        for i in shuffle_target_indices:
            self.triplets[i]['target'] = noise_target.pop()
        logging.info('done')

    def __getitem__(self, index):
        if isinstance(index, slice):
            length = self.__len__()
            start = (0 if index.start is None 
                     else index.start if index.start >= 0 else length + index.start)
            stop = (length if index.stop is None 
                    else index.stop if index.stop >= 0 else length + index.stop)
            start = max(0, min(start, length))
            stop = max(0, min(stop, length))
            step = index.step if index.step else 1
            res = [self.__getitem__(i) for i in range(start, stop, step)]
            return list(zip(*res))
        
        if self.mode == 'relative':
            current_triplet = self.current_triplets[index]
            image_captions = current_triplet['captions']
            reference_name = current_triplet['candidate']
            if self.split in ['train', 'val']:
                target_name = current_triplet['target']
                return reference_name, target_name, image_captions, index
            raise ValueError("split should be in ['train', 'val']")
        elif self.mode == 'gallery':
            image_name = self.current_image_names[index]
            image_path = self.path / 'images' / f"{image_name}.jpg"
            image = self.preprocess(PIL.Image.open(image_path))
            return image_name, image

        raise ValueError("mode should be in ['relative', 'gallery']")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.current_triplets)
        elif self.mode == 'gallery':
            return len(self.current_image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
        
    def load_image_features(self):
        logging.info('load the precomputed features')
        # This is the images with all dress types, in the sequence of "dress, shirt, toptee".
        self.images = torch.load(f'./features/FashionIQ/{self.split}_images.pt')
        with open(f'./features/FashionIQ/{self.split}_name_to_idx.json', 'r') as f:
            self.name_to_idx = json.load(f)
            
    def load_image_features_tensor_and_dict(self, features, dict):
        self.images = features
        self.name_to_idx = dict
    
    def get_image_features(self, name_list):
        indices = [self.name_to_idx[name] for name in name_list]
        images_features = self.images[indices]
        return images_features