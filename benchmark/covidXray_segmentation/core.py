import torch
import ujson
import os
import random
import datetime
import collections
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import metrics
import SimpleITK as sitk
import numpy as np
from benchmark.toolkits import DefaultTaskGen, IDXTaskPipe
from benchmark.toolkits import BasicTaskCalculator
from .utils.utils import get_transform
from utils.loss import ContrastiveLoss


MEAN_STD = {
    'means': 138.65198969,
    'stds': 60.06416719
}


class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=4, skewness=0.5, rawdata_path='/data/zbm/QaTa-COV19-v2', local_hld_rate=0.2, seed=0):
        super(TaskGen, self).__init__(benchmark='covidXray_segmentation',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path=rawdata_path,
                                      local_hld_rate=local_hld_rate,
                                      seed=seed
                                      )
        self.num_classes = 2
        self.save_task = TaskPipe.save_task
        self.visualize = self.visualize_by_class

    def load_data(self):
        test_path = os.path.join(self.rawdata_path, 'Test Set', 'Images')
        self.test_idxs = [fp for fp in os.listdir(test_path)]

    def local_holdout(self, local_datas, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        trainval_path = os.path.join(self.rawdata_path, 'Train Set', 'Images')
        trainval_file = sorted(os.listdir(trainval_path))
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * (1-self.local_holdout_rate))
            train_ids = local_data[:k]
            val_ids = local_data[k:]
            train_name = [trainval_file[id] for id in train_ids]
            val_name = [trainval_file[id] for id in val_ids]
            train_cidxs.append(train_name)
            valid_cidxs.append(val_name)
        return train_cidxs, valid_cidxs

    def partition(self):
        """label_skew_dirichlet"""
        """alpha = (-4log(skewness + epsilon))**4"""
        MIN_ALPHA = 0.01
        alpha = (-4*np.log(self.skewness + 10e-8))**4
        alpha = max(alpha, MIN_ALPHA)

        img_path = os.path.join(self.rawdata_path, 'Train Set', 'Images')
        img_list = sorted(os.listdir(img_path))
        labels = []
        for fp in img_list:
            name = 'mask_' + fp
            img = Image.open(os.path.join(self.rawdata_path, 'Train Set', 'Ground-truths', name)).convert('L')
            img = np.array(img)
            sum_pix = img.shape[0] * img.shape[1]
            img[img < 100] = 0
            img[img > 100] = 1
            sum_mask = np.sum(img)
            ratio = sum_mask / sum_pix
            level = int(ratio / 0.01)
            level = level if level < 49 else 49
            labels.append(level)
        lb_counter = collections.Counter(labels)
        p = np.array([1.0*v/len(labels) for v in lb_counter.values()])

        lb_dict = {}
        labels = np.array(labels)
        for lb in range(len(lb_counter.keys())):
            lb_dict[lb] = np.where(labels == lb)[0]
        proportions = [np.random.dirichlet(alpha*p) for _ in range(self.num_clients)]
        while np.any(np.isnan(proportions)):
            proportions = [np.random.dirichlet(alpha * p) for _ in range(self.num_clients)]
        while True:
            # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
            mean_prop = np.mean(proportions, axis=0)
            error_norm = ((mean_prop-p)**2).sum()
            print("Error: {:.8f}".format(error_norm))
            if error_norm <= 1e-2/self.num_classes:
                break
            exclude_norms = []
            for cid in range(self.num_clients):
                mean_excid = (mean_prop*self.num_clients - proportions[cid])/(self.num_clients-1)
                error_excid = ((mean_excid-p)**2).sum()
                exclude_norms.append(error_excid)
            excid = np.argmin(exclude_norms)
            sup_prop = [np.random.dirichlet(alpha*p) for _ in range(self.num_clients)]
            alter_norms = []
            for cid in range(self.num_clients):
                if np.any(np.isnan(sup_prop[cid])):
                    continue
                mean_alter_cid = mean_prop - proportions[excid]/self.num_clients + sup_prop[cid]/self.num_clients
                error_alter = ((mean_alter_cid-p)**2).sum()
                alter_norms.append(error_alter)
            if len(alter_norms) > 0:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]
        local_datas = [[] for _ in range(self.num_clients)]
        self.dirichlet_dist = []  # for efficiently visualizing
        for lb in lb_counter.keys():
            lb_idxs = lb_dict[lb]
            lb_proportion = np.array([pi[lb] for pi in proportions])
            lb_proportion = lb_proportion/lb_proportion.sum()
            lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_proportion)
            self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
            local_datas = [local_data+lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
        self.dirichlet_dist = np.array(self.dirichlet_dist).T
        for i in range(self.num_clients):
            np.random.shuffle(local_datas[i])

        return local_datas


class TaskCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_no_reduction = torch.nn.CrossEntropyLoss(reduction='none')
        self.DataLoader = DataLoader
        self.contrastive_loss = ContrastiveLoss()

    def train_one_step(self, model, data, labeled_model, image_prototype, contrastive, weight_contrastive=1, islabeled=True):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        loss_consistency = None
        loss_contrastive = None
        tdata = self.data_to_device(data)
        if islabeled:
            outputs = model(tdata[0])
            loss = self.criterion(outputs[0], tdata[-1].long())
        else:
            weak, strong = tdata[0]
            out_weak = model(weak)[0]
            out_strong, features = model(strong)
            pseudo = out_weak.max(1)[1].long()
            if labeled_model is None:
                pseudo_label = pseudo
                mask = torch.ones_like(pseudo_label).long().to(self.device)
            else:
                with torch.no_grad():
                    pseudo_aux = labeled_model(weak)[0].max(1)[1].long()
                mask = (pseudo == pseudo_aux).long()
                pseudo_label = (pseudo + pseudo_aux == 2).long()
            loss_consistency = self.criterion_no_reduction(out_strong, pseudo_label) * mask
            loss_consistency = loss_consistency.sum() / mask.sum()

            if image_prototype is None or not contrastive:
                loss_contrastive = torch.tensor(float('nan'))
            else:
                loss_contrastive = self.contrastive_loss(features[0], pseudo_label, image_prototype, self.device)

            if torch.isnan(loss_contrastive):
                loss = loss_consistency
            else:
                loss = loss_consistency + weight_contrastive*loss_contrastive

        return {'loss': loss, 'loss_consistency': loss_consistency, 'loss_contrastive': loss_contrastive}

    @torch.no_grad()
    def test(self, model, dataset, batch_size=8, num_workers=0, vis=False):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """

        if vis:
            path = '/home/zbm/desktop/data/'
            current_time = datetime.datetime.now()
            path = path + str(current_time)
            os.makedirs(path)

        model.eval()
        if batch_size == -1:
            batch_size = len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        total_loss = total_acc = total_dice = total_sp = total_se = total_js = total_auc = total_hd = 0.0
        hd_length = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            if isinstance(batch_data[0], list):
                inputs = batch_data[0][0]
            else:
                inputs = batch_data[0]

            if vis:
                vis_input = (inputs[0] * MEAN_STD['stds'] + MEAN_STD['means']).type(torch.uint8).cpu().detach().numpy()

            outputs = model(inputs)
            target = batch_data[-1].long()
            batch_mean_loss = self.criterion(outputs[0], target).item()
            y_pred = outputs[0].data.max(1)[1]

            if vis:
                vis_pred = (y_pred[0] * 255).type(torch.uint8).cpu().detach().numpy()
                vis_gt = (target[0] * 255).type(torch.uint8).cpu().detach().numpy()

            tp = (y_pred + target.data.view_as(y_pred) == 2).long().cpu().sum().item()
            tn = (y_pred + target.data.view_as(y_pred) == 0).long().cpu().sum().item()
            fp = ((y_pred == 1).float() + (target.data.view_as(y_pred) == 0) == 2).long().cpu().sum().item()
            fn = ((y_pred == 0).float() + (target.data.view_as(y_pred) == 1) == 2).long().cpu().sum().item()

            dc = 2 * tp / (2 * tp + fn + fp + 1e-6)
            acc = (tp + tn) / (tp + fp + tn + fn + 1e-6)
            se = tp / (tp + fn + 1e-6)
            sp = tn / (tn + fp + 1e-6)
            js = tp / (tp + fn + fp + 1e-6)

            total_dice += dc
            total_acc += acc
            total_se += se
            total_sp += sp
            total_js += js

            if vis:
                pil_input = Image.fromarray(vis_input.transpose(1, 2, 0)).convert('RGB')
                pil_input.save(path + '/' + str(batch_id) + '_img.png')
                pil_pred = Image.fromarray(vis_pred).convert('L')
                pil_pred.save(path + '/' + str(batch_id) + '_mask_' + '%.4f' % dc + '.png')
                pil_gt = Image.fromarray(vis_gt).convert('L')
                pil_gt.save(path + '/' + str(batch_id) + '_gt.png')

                vis_gt[vis_gt < 128] = 0
                vis_gt[vis_gt > 128] = 1
                vis_pred[vis_pred < 128] = 0
                vis_pred[vis_pred > 128] = 1

                vis_out = np.zeros((vis_pred.shape[0], vis_pred.shape[1])).astype(np.int)
                vis_out[(vis_gt == 1) & (vis_pred == 1)] = 3
                vis_out[(vis_gt == 1) & (vis_pred == 0)] = 2
                vis_out[(vis_gt == 0) & (vis_pred == 1)] = 1
                vis_out[(vis_gt == 0) & (vis_pred == 0)] = 0
                color = np.array([
                    [0, 0, 0],
                    [0, 255, 0],  # green
                    [255, 0, 0],  # red
                    [255, 255, 0]]  # yellow
                )
                vis_out = color[vis_out]
                pil_color = Image.fromarray(np.uint8(vis_out)).convert('RGB')
                pil_color.save(path + '/' + str(batch_id) + '_color_' + '%.4f' % dc + '.png')

            if outputs[0].shape[1] == 1:
                score = outputs[0]
            else:
                score = F.softmax(outputs[0], 1)[:, 1, :, :]
            if torch.any(torch.isnan(score)) or torch.any(torch.isnan(target)):
                continue
            fpr, tpr, _ = metrics.roc_curve(target.flatten().cpu().detach().numpy().astype(np.uint8),
                                            score.flatten().cpu().detach().numpy(), pos_label=1)
            total_auc += metrics.auc(fpr, tpr)

            for i in range(target.shape[0]):
                if target[i].sum() != 0 and y_pred[i].sum() != 0:
                    gt = sitk.GetImageFromArray(target[i].cpu(), isVector=False)
                    my_mask = sitk.GetImageFromArray(y_pred[i].cpu(), isVector=False)
                    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
                    hausdorff_distance_filter.Execute(gt, my_mask)
                    total_hd += hausdorff_distance_filter.GetHausdorffDistance()
                    hd_length += 1

            total_loss += batch_mean_loss * len(target)

        loss = total_loss/len(dataset)
        accuracy = total_acc/len(dataset)
        sensitivity = total_se/len(dataset)
        specificity = total_sp/len(dataset)
        dice = total_dice/len(dataset)
        jaccard_similarity = total_js/len(dataset)
        auc = total_auc/len(dataset)
        hd95 = -1 if hd_length == 0 else total_hd/hd_length

        return {'accuracy': accuracy, 'sensitivity': sensitivity, 'specificity': specificity, 'jaccard_similarity': jaccard_similarity, 'Dice': dice, 'AUC': auc, 'HD95': hd95, 'loss': loss}

    @torch.no_grad()
    def get_prototypes(self, model, labeled_model, dataset, batch_size=1, num_workers=0):
        model.eval()
        fore_protptypes = []
        back_prototypes = []
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            if isinstance(batch_data[0], list):
                weak, strong = batch_data[0]
                out_weak = model(weak)[0]
                pseudo = out_weak.max(1)[1].long()
                if labeled_model is None:
                    mask_fore = (pseudo == 1).long()
                    mask_back = (pseudo == 0).long()
                else:
                    pseudo_aux = labeled_model(weak)[0].max(1)[1].long()
                    mask_fore = (pseudo + pseudo_aux == 2).long()
                    mask_back = (pseudo + pseudo_aux == 0).long()
            else:
                mask_fore = batch_data[1]
                mask_back = 1 - mask_fore
                strong = batch_data[0]
            out_feature = model(strong)[1][0]
            b, c, w, h = out_feature.shape
            mask_fore = F.adaptive_avg_pool2d(mask_fore.float(), (w, h)) > 0.5
            mask_back = F.adaptive_avg_pool2d(mask_back.float(), (w, h)) > 0.5
            out_feature = out_feature.permute(0, 2, 3, 1)
            mask_fore = mask_fore.unsqueeze(3).repeat(1, 1, 1, c)
            mask_back = mask_back.unsqueeze(3).repeat(1, 1, 1, c)
            fore_protptype = torch.masked_select(out_feature, mask_fore).view(-1, c).mean(0)
            back_protptype = torch.masked_select(out_feature, mask_back).view(-1, c).mean(0)
            if not torch.isnan(fore_protptype[0]):
                fore_protptypes.append(fore_protptype)
            if not torch.isnan(back_protptype[0]):
                back_prototypes.append(back_protptype)
        return fore_protptypes, back_prototypes

    @torch.no_grad()
    def get_consistency(self, model, labeled_model, dataset, batch_size=1, num_workers=0):
        if labeled_model is None:
            return 0
        model.eval()
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        sum_pixel = 0
        con_pixel = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            if not isinstance(batch_data[0], list):
                return torch.ones(1).to(self.device)
            weak, strong = batch_data[0]
            out_weak = model(weak)[0]
            pseudo_aux = labeled_model(weak)[0].max(1)[1].long()
            pseudo = out_weak.max(1)[1].long()
            mask = (pseudo == pseudo_aux).long()
            sum_pixel += pseudo.view(-1).shape[0]
            con_pixel += mask.sum()

        return (con_pixel / sum_pixel).item()

    @torch.no_grad()
    def get_entropy(self, model, dataset, batch_size=1, num_workers=0):
        model.eval()
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        length = 0
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            if isinstance(batch_data[0], list):
                weak, strong = batch_data[0]
                input_value = weak
            else:
                input_value = batch_data[0]
            out = model(input_value)[0]
            out = F.softmax(out, 1)
            out = - out * out.log()
            out = out.nanmean()
            length += 1
            if batch_id == 0:
                entropy = out
            else:
                entropy += out

        entropy = entropy / length

        return entropy

    def data_to_device(self, data):
        if isinstance(data[0], list):
            return [data[0][0].to(self.device), data[0][1].to(self.device)], data[1].to(self.device)
        return data[0].to(self.device), data[1].to(self.device)

    def get_data_loader(self, dataset, batch_size=4, shuffle=True, num_workers=0):
        if self.DataLoader is None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class TaskPipe(IDXTaskPipe):
    class TaskDataset(torch.utils.data.Dataset):
        def __init__(self, indexs, cid=-1, labeled=True, test=False):
            self.labeled = labeled
            self.test = test
            self.dataidx = cid
            self.path = '/data/zbm/QaTa-COV19-v2/'
            self.file_indexs = indexs
            self.data, self.target = self.__build_dataset__()
            self.transform_labeled, self.transform_unlabeled, self.transform_test = get_transform()

        def __build_dataset__(self):
            split_name = 'Test Set' if self.dataidx == -1 else 'Train Set'

            img_path = os.path.join(self.path, split_name, 'Images')
            mask_path = os.path.join(self.path, split_name, 'Ground-truths')

            img_list = [os.path.join(img_path, i) for i in self.file_indexs]
            mask_list = [os.path.join(mask_path, 'mask_' + i) for i in self.file_indexs]

            assert len(img_list) == len(mask_list)

            return img_list, mask_list

        def __getitem__(self, index):
            img, target = self.data[index], self.target[index]

            img = Image.open(img).convert("RGB")
            target = Image.open(target).convert("L")

            if self.test:
                img, target = self.transform_test(img, target)
                target = target.squeeze(0)
            elif self.labeled:
                img, target = self.transform_labeled(img, target)
                target = target.squeeze(0)
            else:
                img, target = self.transform_unlabeled.weak(img, target)
                img_s = self.transform_unlabeled.strong(img)
                img_s = self.transform_unlabeled.normalize(img_s)
                img_w = self.transform_unlabeled.normalize(img)
                target = self.transform_unlabeled.tensor(target)
                target = target.squeeze(0)
                img = [img_w, img_s]

            return img, target

        def __len__(self):
            return len(self.data)

    @classmethod
    def load_task(cls, task_path, labeledclients):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        cnames = feddata['client_names']
        labeledclients = [True if c in labeledclients else False for c in range(len(cnames))]
        train_datas = [cls.TaskDataset(feddata[cnames[cid]]['dtrain'], cid, labeledclients[cid]) for cid in range(len(cnames))]
        valid_datas = [cls.TaskDataset(feddata[cnames[cid]]['dvalid'], cid) for cid in range(len(cnames))]
        test_data = cls.TaskDataset(feddata['dtest'], test=True)
        return train_datas, valid_datas, test_data, cnames

    @classmethod
    def save_task(cls, generator):
        feddata = {
            'store': 'IDX',
            'client_names': generator.cnames,
            'data_path': generator.rawdata_path
        }
        for cid in range(len(generator.cnames)):
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid],
            }
        feddata['dtest'] = generator.test_idxs
        with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
