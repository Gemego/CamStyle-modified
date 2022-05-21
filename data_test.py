from data import DatasetDataLoader

if __name__ == '__main__':
    data_loader = DatasetDataLoader()
    dataset_size = len(data_loader)
    print('training images = %d' % dataset_size)

    for i, data in enumerate(data_loader):
        print(i, data)
