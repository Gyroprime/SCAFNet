import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
from datasets.data_utils import denormalize




class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader
        self.data_name = args.data_name
        self.n_class = args.n_class

        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)


        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)


        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)



        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir


        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')

            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'],strict=False)

            self.net_G.to(self.device)


            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):

        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)

        pred_vis = pred * 255

        pred_img = np.stack([self.pred, self.pred, self.pred], axis=1)
        h = self.pred.shape[1]
        w = self.pred.shape[2]


        pred_img = pred_img[0,:,:,:]

        for i in range(h):
            for j in range(w):
                if self.pred[0][i][j] == 1 and self.gt[0][0][i][j] == 0:
                    pred_img[0][i][j] = 1
                    pred_img[1][i][j] = 0
                    pred_img[2][i][j] = 0
                elif self.pred[0][i][j] == 0 and self.gt[0][0][i][j] == 1:
                    pred_img[0][i][j] = 0
                    pred_img[1][i][j] = 1
                    pred_img[2][i][j] = 0
        pred_img = pred_img * 255
        pred_img = pred_img.transpose(1, 2, 0)
        pred_img = pred_img.astype(np.uint8)
        return pred_vis, pred_img



    def _update_metric(self):

        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        self.pred = G_pred.cpu().numpy()
        self.gt = target.cpu().numpy()
        current_score = self.running_metric.update_cm(pr=self.pred, gt=self.gt)
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        vis_input = utils.make_numpy_grid(denormalize(self.batch['A'], self.data_name,'A'))
        vis_input2 = utils.make_numpy_grid(denormalize(self.batch['B'],self.data_name,'B'))
        pred_vis, pred_colorimg = self._visualize_pred()
        vis_pred = utils.make_numpy_grid(pred_vis)
        vis_gt = utils.make_numpy_grid(self.batch['L'])
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id) + '.jpg')
        color_file_name = os.path.join(
            self.vis_dir, 'color_eval_' + str(self.batch_id) + '.jpg')
        plt.imsave(color_file_name, pred_colorimg)
        plt.imsave(file_name, vis);


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)


    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()


        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
