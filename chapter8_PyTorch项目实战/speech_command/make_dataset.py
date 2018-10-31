import argparse
import os
import shutil


# 把文件从源文件夹移动到目标文件夹
def move_files(original_fold, data_fold, data_filename):
    with open(data_filename) as f:
        for line in f.readlines():
            vals = line.split('/')
            dest_fold = os.path.join(data_fold, vals[0])
            if not os.path.exists(dest_fold):
                os.mkdir(dest_fold)
            shutil.move(os.path.join(original_fold, line[:-1]), os.path.join(data_fold, line[:-1]))


# 建立train文件夹
def create_train_fold(original_fold, train_fold, test_fold):
    # 文件夹名列表
    dir_names = list()
    for file in os.listdir(test_fold):
        if os.path.isdir(os.path.join(test_fold, file)):
            dir_names.append(file)

    # 建立训练文件夹train
    for file in os.listdir(original_fold):
        if os.path.isdir(os.path.join(test_fold, file)) and file in dir_names:
            shutil.move(os.path.join(original_fold, file), os.path.join(train_fold, file))


# 建立数据集,train,valid, 和　test
def make_dataset(in_path, out_path):
    validation_path = os.path.join(in_path, 'validation_list.txt')
    test_path = os.path.join(in_path, 'testing_list.txt')

    # train, valid, test三个数据集文件夹的建立
    train_fold = os.path.join(out_path, 'train')
    valid_fold = os.path.join(out_path, 'valid')
    test_fold = os.path.join(out_path, 'test')

    for fold in [valid_fold, test_fold, train_fold]:
        if not os.path.exists(fold):
            os.mkdir(fold)
    # 移动train, valid, test三个数据集所需要的文件
    move_files(in_path, test_fold, test_path)
    move_files(in_path, valid_fold, validation_path)
    create_train_fold(in_path, train_fold, test_fold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make speech commands dataset.')
    parser.add_argument('--in_path', default='org_data',
                        help='the path to the root folder of te speech commands dataset.')
    parser.add_argument('--out_path', default='data', help='the path where to save the files splitted to folders.')
    args = parser.parse_args()
    make_dataset(args.in_path, args.out_path)
