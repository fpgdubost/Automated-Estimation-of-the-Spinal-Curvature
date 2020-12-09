# coding=utf-8
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from matplotlib import pyplot as plt
import csv
from keras import callbacks


class LossHistory(callbacks.Callback):

    def __init__(self, savePath, loss='loss', batch_size=1, padding=0):

        # parameters of the csv and plot
        self.reports = [{'name': 'evolution',
                         'type': 'text',
                         'file': 'evolution.csv',
                         'vars': ['loss', 'val_loss']},

                        {'name': 'losses plot',
                         'type': 'plot',
                         'file': 'losses.png',
                         'unity': loss,
                         'vars': ['loss', 'val_loss']}
                        ]

        self.vars_name = ['loss', 'val_loss']

        self.savePath = savePath

        self.batch_size = batch_size

        self.padding = padding

        self.vars = {}
        self.extremum = {}

        for v in self.vars_name:
            self.vars[v] = []
            self.extremum[v] = 0


    def on_epoch_end(self, epoch, logs={}):
        """
            Params : logs : a dictionnary that contains the vars that we want to write in csv or plot
        """
        # if epoch < 20:
        #     if epoch % 2 == 0:
        #         validation_results = self.model.predict(self.validation_data[0], batch_size=self.batch_size)
        #         self.plot_img(self.validation_data[0], self.validation_data[1], validation_results, epoch)
        # else:
        #     if epoch % 10 == 0:
        #         validation_results = self.model.predict(self.validation_data[0], batch_size=self.batch_size)
        #         self.plot_img(self.validation_data[0], self.validation_data[1], validation_results, epoch)

        if epoch == 0:
            toDelete = []

        for v in self.vars:
            if v in logs.keys():
                if v == 'fnr_val' and logs[v] > 1:
                    self.vars[v].append(1)
                else:
                    self.vars[v].append(logs[v])
            elif epoch == 0:
                toDelete.append(v)

        if epoch == 0:
            for v in toDelete:
                print 'var ' + v + ' not found\ndelete...'
                del self.vars[v]

        for r in self.reports:
            if r['type'] == 'text':
                self.writeCSV(r, logs, True if epoch == 0 else False)
            elif r['type'] == 'plot':
                self.plot(r, logs)


    def writeCSV(self, report, logs, rewrite=False):

        with open(os.path.join(self.savePath, 'evolution.csv'), "w" if rewrite else "a") as myfile:
            writer = csv.writer(myfile, delimiter=';')

            if rewrite:
                writer.writerow([v_name for v_name in report['vars']])

            print(report['vars'])
            writer.writerow([self.vars[v_name][-1] for v_name in report['vars']])

    def plot(self, report, logs):

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
        lineStyle = ['dashed', 'solid', 'dotted']

        plt.figure(figsize=(10, 12))

        plt.title(report['name'])
        plt.ylabel(report['unity'])
        plt.xlabel(u'epoch')

        # plot each variable
        for i, v_name in enumerate(report['vars']):
            plt.plot(range(0, len(self.vars[v_name])), self.vars[v_name], ls=lineStyle[1],
                     color=colors[i], label=u'{}'.format(v_name))

        # Display the legend
        lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=20)
        plt.grid('on')

        try:
            plt.savefig(os.path.join(self.savePath, report['file']), bbox_extra_artists=(lgd,), bbox_inches='tight')
        except Exception as inst:
            print(type(inst))
            print(inst)
        plt.close()


    def plot_img(self, datas, GTs, results, epoch):
        print("\nPlotting result...")
        nb_image = 3
        step = datas.shape[0] / nb_image
        if epoch == 0:
            os.makedirs(os.path.join(self.savePath, 'img_callback'))

        for k in range(nb_image):

            # select random index
            index = k * step
            if epoch == 0:
                os.makedirs(os.path.join(self.savePath, 'img_callback', str(index)))

            # print original images and the result of augmentation
            data = datas[index]
            GT = GTs[index]
            result = results[index]

            steps = np.linspace(0, GT.shape[2] - 1 - self.padding, num=10, dtype=np.int)

            plt.figure(figsize=(16, 12), dpi=400)
            plt.suptitle(u'plot tile n°' + repr(index), fontsize=20)
            # print(X[1])
            bias = 0
            for i in range(10):

                im = data[:, :, self.padding + steps[i], 0]
                ax = plt.subplot(5, 7, 3 * i + 1 + bias)
                plt.imshow(np.squeeze(im), cmap='gray')  # , vmin=0, vmax=1)
                plt.axis(u'off')
                pltname = u'data slice ' + str(steps[i] + self.padding)
                fz = 5  # Works best after saving
                ax.set_title(pltname, fontsize=fz)

                colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'greenyellow', 'gold']
                cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
                for j in range(GT.shape[3]):
                    cmap = plt.get_cmap(cmaps[j])

                    gt = GT[:, :, steps[i], j]

                    im = cmap(gt)
                    im[..., -1] = gt
                    ax = plt.subplot(5, 7, 3 * i + 2 + bias)
                    plt.imshow(im)  # , vmin=0, vmax=1)
                    plt.axis(u'off')
                    pltname = u'GT slice ' + str(steps[i] + self.padding)
                    fz = 5  # Works best after saving
                    ax.set_title(pltname, fontsize=fz)

                    res = result[:, :, steps[i], j]
                    im = cmap(res)
                    im[..., -1] = res
                    ax = plt.subplot(5, 7, 3 * i + 3 + bias)
                    plt.imshow(im)  # , vmin=0, vmax=1)
                    plt.axis(u'off')
                    pltname = u'result slice ' + str(steps[i] + self.padding)
                    fz = 5  # Works best after saving
                    ax.set_title(pltname, fontsize=fz)

                if i % 2 == 0:
                    bias += 1
            plt.savefig(
                os.path.join(self.savePath, 'img_callback', str(index), str(epoch) + '_im.png'))
            plt.close()


class saveEveryNModels(callbacks.Callback):
    def __init__(self, savePath, period):
        super(callbacks.Callback, self).__init__()
        self.savePath = savePath
        self.period = period

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.period == 0 and epoch != 0:
            if not os.path.exists(os.path.join(self.savePath, 'weights')):
                os.makedirs(os.path.join(self.savePath, 'weights'))
            self.model.save_weights(os.path.join(self.savePath, 'weights', 'best_weights_' + str(epoch) + '.hdf5'),
                                    overwrite=True)
