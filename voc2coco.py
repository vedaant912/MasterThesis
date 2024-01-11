import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re


def get_label2id(labels_path: list) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = f.read().split()
    labels_ids = list(range(0, len(labels_str)))
    return dict(zip(labels_str, labels_ids))

def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:
    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    with open(ann_ids_path, 'r') as f:
        ann_ids = f.read().split()
    ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(a_path, extract_num_from_imgid=True):
    
    file_name = a_path.split('/')[3]

    print(file_name)
    if extract_num_from_imgid:
        img_id = file_name.split('.')[0].split('_')[2]
    
    image_info = {
        'file_name': file_name.split('.')[0] + '.png',
        'height': 512,
        'width': 512,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(box, label, label2id):


    xmin = box[0][0]
    # xmax = right corner x-coordinates
    xmax = box[1][0]
    # ymin = left corner y-coordinates
    ymin = box[0][1]
    # ymax = right corner y-coordinates
    ymax = box[1][1]

    category_id = label

    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    
    o_width = xmax - xmin
    o_height = ymax - ymin

    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    
    for a_path in tqdm(annotation_paths):

        img_info = get_image_info(a_path, extract_num_from_imgid=True)

        img_id = img_info['id']

        output_json_dict['images'].append(img_info)

        with open('./input/test_txts/'+img_info['file_name'].split('.')[0]+'.txt', 'r') as file:
            data = json.load(file)
            
        bboxes = data['bboxes']
        labels = data['pedestrian_class']

        for box,label in zip(bboxes,labels):

            ann = get_coco_annotation_from_obj(box=box, label=label, label2id=label2id)
        
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
    
    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=None,
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str, default=None,
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
    parser.add_argument('--labels', type=str, default=None,
                        help='path to label list.')
    parser.add_argument('--output', type=str, default='test.json', help='path to output json file')
    parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
    parser.add_argument('--extract_num_from_imgid', action="store_true",
                        help='Extract image number from the image filename')
    args = parser.parse_args()
    
    file_list = os.listdir('./input/test_txts/')

    label2id = get_label2id(labels_path='./labels.txt')
    print(label2id)


    # ann_paths = get_annpaths(
    #     ann_dir_path=args.ann_dir,
    #     ann_ids_path=args.ann_ids,
    #     ext=args.ext,
    #     annpaths_list_path=args.ann_paths_list
    # )
    ann_paths = ['./input/test_txts/' + file for file in file_list]

    print(ann_paths)


    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=args.extract_num_from_imgid
    )


if __name__ == '__main__':
    main()
