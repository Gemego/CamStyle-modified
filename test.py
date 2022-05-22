from data import DatasetDataLoader
from models.gan_setup import create_model

if __name__ == '__main__':
    # 前面设置了一些参数还需要看一下，这个和train逐个对比了一下，只要test和train不冲突的都可以不加
    data_loader = DatasetDataLoader(isTrain=False)  # 设置为不训练
    dataset = data_loader.load_data()
    model = create_model(False, 'market-c1-c2')
    # 此处的参数需要注意一下，猜测第一个是istrain,name后续再看看应该谁的
    model.setup()
    for i, data in enumerate(dataset):  # inner loop within one epoch
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        # save_images(visuals, img_path, opt.camA, opt.camB, opt.save_root)
