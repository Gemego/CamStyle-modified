import numpy as np
import torch
import os
import ntpath
import time
from PIL import Image
import errno


# mkdir
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


# save image to the disk
def save_images(visuals, image_path, camA=1, camB=2, save_root=None):
    mkdir_if_missing(save_root)
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    ims = []
    txts = []
    links = []

    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        if label in ('rec_A', 'rec_B', 'real_A', 'real_B'):
            continue
        import re
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid, cam = map(int, pattern.search(name).groups())
        print(name, cam)

        if not (cam == camA and label == 'fake_B') and not (cam == camB and label == 'fake_A'):
            continue
        if label == 'fake_B':
            label = 'fake_' + str(camA) + 'to' + str(camB)
        elif label == 'fake_A':
            label = 'fake_' + str(camB) + 'to' + str(camA)

        image_name = '%s_%s.jpg' % (name, label)
        save_path = os.path.join(save_root, image_name)
        save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)


class Visualizer:
    def __init__(self, isTrain=True):
        self.display_id = 1
        self.use_html = isTrain
        self.win_size = 256
        self.name = 'experiment_name'
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.ncols = 0
            self.vis = visdom.Visdom(server="http://localhost", port=8097)

        if self.use_html:
            self.web_dir = os.path.join('./checkpoints', 'experiment_name', 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join('./checkpoints', 'experiment_name', 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except ConnectionError:
                    print('Could not connect to Visdom server.')
                    exit(1)

            else:
                idx = 1
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
