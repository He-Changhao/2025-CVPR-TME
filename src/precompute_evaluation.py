import torch
import numpy as np
from operator import itemgetter
from utility import device
import logging
from tqdm import tqdm
from statistics import mean
import data

def evaluate_features(model, dataset, text_preprocessor):
    acc_dict = {}
    dataset_name = 'CIRR' if isinstance(dataset, data.CIRR) else 'FashionIQ'
    if dataset_name == 'FashionIQ':
        acc_dict['acc'] = 0
        for category in ['dress', 'shirt', 'toptee']:
            dataset.set_specific_dress_type(category)
            acc_dict[category] = compute_metric_features(model, dataset, text_preprocessor, category=category)
            acc_dict['acc'] += acc_dict[category]['acc']
        acc_dict['acc'] /= 3
        acc_dict['mean'] = acc_dict['acc']
        torch.cuda.empty_cache()
    else:
        acc_dict = compute_metric_features(model, dataset, text_preprocessor, category=dataset_name)
    logging.info(f"current metric acc: {acc_dict['acc']}")
    return acc_dict

def compute_metric_features(model, dataset, text_preprocessor, category):
    """
    Compute the metric results for the given dataset and model.
    If you find cuda out of memory, please try to reduce the batch size, that is Qformer_bs and query_bs.
    Args:
        model (torch.nn.Module): The model to compute the metric features.
        dataset (torch.utils.data.Dataset): The dataset to compute the metric features.
        text_preprocessor (Callable): The text preprocessor to preprocess the text.
        category (str): The category of the dataset.
    Returns:
        dict: The metric results.
    """
    Qformer_bs = 256
    n_images = len(dataset.current_image_names)
    F_images = []
    print('Evaluation for', category)
    print('compute image last hidden states')
    print('num_targets: ', n_images, end='\t')
    print('num_queries: ', len(dataset))
    for idx in tqdm(range(0, n_images, Qformer_bs), ncols=120, mininterval=30):
        cur_image_embed = dataset.get_image_features(dataset.current_image_names[idx:idx+Qformer_bs]).to(device)
        with torch.no_grad():
            F_image = model.encode_image(cur_image_embed)
        F_images.append(F_image)
    F_images = torch.cat(F_images, dim=0)
    name_to_idx = {name: i for i, name in enumerate(dataset.current_image_names)}
    index_names = dataset.current_image_names
    target_names = []
    reference_names = []
    distances = []
    subset_names = []
    print('Compute simlarity: ')
    query_bs = 32
    for idx in tqdm(range(0, len(dataset), query_bs), ncols=120, mininterval=30):
        batch = dataset[idx:idx+query_bs]
        if category == 'CIRR':
            reference_name, target_name, captions, group_members = batch
        else:
            reference_name, target_name, captions, _ = batch
        r_indices = [name_to_idx[name] for name in reference_name]
        F_r = F_images[r_indices]
        if category != 'CIRR':
            captions = np.array(captions).flatten().tolist()
            captions = [f"{captions[i].strip('.?, ').capitalize()} and {captions[i + 1].strip('.?, ')}" for i in range(0, len(captions), 2)]
        captions = [text_preprocessor(item) for item in captions]
        distance:torch.Tensor = model.inference(F_r, F_images, captions)
        # if you find cuda out of memory: distance = distance.to('cpu')
        distances.append(distance)
        target_names.extend(target_name)
        reference_names.extend(reference_name)
        if category == 'CIRR':
            subset_names.extend(group_members)

    distances = torch.vstack(distances)
    distances = 1 - distances
    ref_images_indices_in_index_names = []
    if category == 'CIRR': # only for cirr, delete the reference image from current query.
        for i in reference_names:
            ref_images_indices_in_index_names.append(index_names.index(i))
        distances[list(range(distances.shape[0])), ref_images_indices_in_index_names] = 10e10
    
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names))
        .reshape(len(target_names), -1)
        )
    
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    
    acc_dict = {
        'recall_at1': recall_at1,
        'recall_at5': recall_at5,
        'recall_at10': recall_at10,
        'recall_at50': recall_at50,
        'acc': (recall_at10 + recall_at50) / 2.,
        'mean': mean([recall_at10, recall_at50])
    }
    
    # Compute the subset predictions and ground-truth labels
    if category == 'CIRR':
        subset_names = np.array(subset_names)
        group_mask = (sorted_index_names[..., None] == subset_names[:, None, :]).sum(-1).astype(bool)
        group_labels = labels[group_mask].reshape(labels.shape[0], -1)
        
        group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
        group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
        group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100
        acc_dict.update({
            'group_recall_at1': group_recall_at1, 
            'group_recall_at2': group_recall_at2,
            'group_recall_at3': group_recall_at3,
            'acc': (group_recall_at1 + recall_at5) / 2.,
            'mean': mean([recall_at1, recall_at5, recall_at10, recall_at50, 
                                        group_recall_at1, group_recall_at2, group_recall_at3])
        })
    
    if category == 'CIRR':
        logging.info(f'   CIRR      R@1: {recall_at1}, R@5: {recall_at5}, R@10: {recall_at10}, R@50: {recall_at50}')
        logging.info(f'CIRR subset  R@1: {group_recall_at1}, R@2: {group_recall_at2}, R@3: {group_recall_at3}')
        logging.info(f"Avg(R@5+R@1): {round(acc_dict['acc'], 3)}, mean: {round(acc_dict['mean'], 3)}")
    else:
        logging.info(f"{category}   R@1: {recall_at1}, R@10: {recall_at10}, R@50: {recall_at50}, Avg(R@10,R@50): {round(acc_dict['acc'],3)}")        
    return acc_dict
    
