import time
from options.train_options import TrainOptions
from data import DatasetDataLoader
from models.gan_setup import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    data_loader = DatasetDataLoader()
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(True, 'market-c1-c2')
    model.setup()
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(1, 51):    # outer loop for different epochs
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        t_data = 0
        for i, data in enumerate(dataset):    # inner loop within one epoch
            iter_start_time = time.time()
            if total_steps % 100 == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += 1
            epoch_iter += 1
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % 400 == 0:     # display images on visdom and save images to a HTML file
                save_result = total_steps % 1000 == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % 100 == 0:      # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / 1
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if 1 > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % 5000 == 0:     # cache the latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % 10 == 0:      # cache the model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, 50, time.time() - epoch_start_time))
        model.update_learning_rate()    # update learning rates at the end of every epoch.
