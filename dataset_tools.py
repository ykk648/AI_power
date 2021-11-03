from dataset_preprocess import count_mean_std, gen_txt_from_path

"""
generate dataset list as follow formats:
|-- data
    |-- train
        |--label1
            |--dataset prefix 1
                |--*.jpg
            |--dataset prefix 2
                |--*.jpg
        |--label2
            |--dataset prefix
                |--*.jpg
        |--label
            |--dataset prefix
                |--*.jpg
        ...

    |-- val
        |--*.jpg
    |--train.txt
    |--val.txt
"""

if __name__ == '__main__':
    test_path = ''
    # count_mean_std(test_path)
    gen_txt_from_path(test_path, img_format='jpg', train_ratio=0.8)
