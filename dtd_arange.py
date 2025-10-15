import os
import shutil


def create_directory_structure(base_dir, classes, split='1'):
    target_dir = {'train': 'train', 'val': 'train', 'test': 'val'}
    for dataset in ['train', 'val', 'test']:
        path = os.path.join(base_dir, target_dir[dataset])
        os.makedirs(path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(path, cls), exist_ok=True)

        split_file = f'{dataset}{split}.txt'
        with open(os.path.join(base_dir, 'labels', split_file), 'r') as f:
            for line in f:
                line = line.strip()
                src_path = os.path.join(base_dir, 'images', line)
                dst_path = os.path.join(base_dir, target_dir[dataset], line)
                print(src_path, dst_path)
                shutil.copy(src_path, dst_path)


# Replace with the absolute path to your dataset
base_dir = '/home/junghwan/task_singular_vectors/datasets/dtd'

classes = [d for d in os.listdir(os.path.join(
    base_dir, 'images')) if os.path.isdir(os.path.join(base_dir, 'images', d))]

create_directory_structure(base_dir, classes)
