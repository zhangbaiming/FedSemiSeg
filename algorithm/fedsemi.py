import copy
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import utils.systemic_simulator as ss
import collections
from sklearn.cluster import k_means
from utils import fmodule
from .fedbase import BasicServer, BasicClient


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.labeled_models = [None for _ in range(self.num_clients)]
        self.image_prototypes = None
        self.agg_label_mode = option['aggregate_labeled']
        self.agg_coefficient = option['agg_coefficient']
        self.task = option['task']
        self.normal_images = torch.randn(option['num_normimg'], 3, 224, 224)
        self.num_normal = option['num_normimg']

    def model_dis(self, m1, m2):
        temp_model = m1 - m2
        dist_total = torch.zeros(1).float().to(self.device)
        for key in temp_model:
            dist = torch.norm(temp_model[key])
            dist_total += dist
        return dist_total

    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if model is None:
            model = self.model
        if self.test_data:
            if 'Xray' not in self.task:
                all_metrics = collections.defaultdict(list)
                for tdata in self.test_data:
                    client_metrics = self.calculator.test(model, tdata, batch_size=self.option['test_batch_size'])
                    for met_name, met_val in client_metrics.items():
                        all_metrics[met_name].append(met_val)
                return all_metrics
            else:
                return self.calculator.test(model, self.test_data, batch_size=self.option['test_batch_size'], vis=self.option['vis'])
        else:
            return None

    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        packages = self.communicate(self.selected_clients)
        models = packages['model']
        client_prototypes = packages['client_prototype']
        image_prototypes = packages['image_prototype']
        agg_statistics = packages['agg_statistics']

        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, agg_statistics)
        self.labeled_models = self.aggregate_labeled_models(models, client_prototypes)
        self.image_prototypes = image_prototypes
        return

    def pack(self, client_id):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        """
        return {
            "model": copy.deepcopy(self.model),
            "labeled_model": self.labeled_models[client_id],
            "image_prototype": self.image_prototypes
        }

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            res: collections.defaultdict that contains several lists of the clients' reply
        """
        res = collections.defaultdict(list)
        for cpkg in packages_received_from_clients:
            for pname, pval in cpkg.items():
                res[pname].append(pval)

        prototypes = res['image_prototype']
        merge_prototypes = [[] for _ in range(len(prototypes[0]))]
        for cp in prototypes:
            for id, mp in enumerate(merge_prototypes):
                if cp[id] is not None:
                    mp.append(cp[id])
        merge_prototypes = [torch.cat(mp, 0) for mp in merge_prototypes]
        res['image_prototype'] = merge_prototypes

        return res

    @ss.with_inactivity
    def sample(self):
        """Sample the clients.
        :param
        :return
            a list of the ids of the selected clients
        """
        all_clients = [cid for cid in range(self.num_clients)]
        # full sampling with unlimited communication resources of the server
        if self.sample_option == 'full':
            return all_clients
        # sample clients
        elif self.sample_option == 'uniform':
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=False))
        elif self.sample_option == 'md':
            # the default setting that is introduced by FedProx, where the clients are sampled with the probability in proportion to their local data sizes
            p = np.array(self.local_data_vols)/self.total_data_vol
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=p))
        elif self.sample_option == 'labeled':
            selected_clients = [cid for cid, client in enumerate(self.clients) if client.islabeled]
        return selected_clients

    def aggregate(self, models: list, agg_statistics):
        """
        Aggregate the locally improved models.
        :param
            models: a list of local models
            p: a list of weights for aggregating
        :return
            the averaged result
        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==========================================================================================================================
        N/K * Σpk * model_k             |1/K * Σmodel_k             |(1-Σpk) * w_old + Σpk * model_k  |Σ(pk/Σpk) * model_k
        """
        self.total_data_vol = sum([self.local_data_vols[cid] for cid in self.selected_clients])
        if len(models) == 0:
            return self.model
        if self.aggregation_option == 'weighted_scale':
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.selected_clients]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
        elif self.aggregation_option == 'weighted_entropy':
            datap = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.selected_clients]
            if agg_statistics is None:
                p = datap
            else:
                optip = [1.0 * agg_statistics[cid] / sum(agg_statistics) for cid in self.selected_clients]
                p = [self.agg_coefficient * datap[cid] + (1-self.agg_coefficient) * optip for cid in self.selected_clients]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
        elif self.aggregation_option == 'weighted_entropy_dis':
            datap = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.selected_clients]
            if agg_statistics is None:
                p = datap
            else:
                expp = [-math.exp(aggs) for aggs in agg_statistics]
                optip = [1.0 * expp[cid] / sum(expp) for cid in self.selected_clients]
                p = [self.agg_coefficient * datap[cid] + (1-self.agg_coefficient) * optip for cid in self.selected_clients]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
        elif self.aggregation_option == 'weighted_consistency':
            if agg_statistics is None:
                p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.selected_clients]
            else:
                correct_data_vols = [self.local_data_vols[cid] if self.clients[cid].islabeled else self.local_data_vols[cid] * agg_statistics[cid] for cid in self.selected_clients]
                p = [1.0 * correct_data_vols[cid] / sum(correct_data_vols) for cid in self.selected_clients]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
        elif self.aggregation_option == 'uniform':
            return fmodule._model_average(models)
        elif self.aggregation_option == 'weighted_com':
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.selected_clients]
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0-sum(p))*self.model + w
        else:
            p = [1.0 * self.local_data_vols[cid] / self.total_data_vol for cid in self.selected_clients]
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])

    def aggregate_labeled_models(self, models: list, prototypes):
        labeled_models = [model for cid, model in enumerate(models) if self.clients[cid].islabeled]
        labeled_prototypes = [prototype for cid, prototype in enumerate(prototypes) if self.clients[cid].islabeled]
        labeled_datavols = [vol for cid, vol in enumerate(self.local_data_vols) if self.clients[cid].islabeled]

        if self.agg_label_mode.startswith('out'):
            outs = []
            with torch.no_grad():
                for model in models:
                    model.eval()
                    outs.append(model(self.normal_images.to(self.device))[0])
            labeled_outs = [model for cid, model in enumerate(outs) if self.clients[cid].islabeled]

        agg_labeled_models = []
        for cid in range(self.num_clients):
            if self.clients[cid].islabeled or (cid >= len(prototypes)) or (prototypes[cid] is None and self.agg_label_mode.startswith('proto')) or self.agg_label_mode == 'none':
                agg_labeled_models.append(None)
            else:
                # 'proto_sim', 'proto_sim_max', 'proto_dis', 'proto_dis_max',
                if self.agg_label_mode == 'proto_sim':
                    labeled_weights = []
                    for labeledid in range(len(labeled_models)):
                        labeled_weights.append(F.cosine_similarity(prototypes[cid], labeled_prototypes[labeledid], dim=0))
                    labeled_weights = [math.exp(labeled_weight) for labeled_weight in labeled_weights]
                    labeled_weights = [w/sum(labeled_weights) for w in labeled_weights]
                    labeled_model = fmodule._model_sum([k * p for k, p in zip(labeled_models, labeled_weights)])
                elif self.agg_label_mode == 'proto_sim_max':
                    labeled_model = labeled_models[0]
                    max_sim = F.cosine_similarity(prototypes[cid], labeled_prototypes[0], dim=0).item()
                    for labeledid in range(1, len(labeled_models)):
                        sim = F.cosine_similarity(prototypes[cid], labeled_prototypes[labeledid], dim=0).item()
                        if max_sim < sim:
                            max_sim = sim
                            labeled_model = labeled_models[labeledid]
                    labeled_model = copy.deepcopy(labeled_model)
                elif self.agg_label_mode == 'proto_dis':
                    labeled_weights = []
                    for labeledid in range(len(labeled_models)):
                        labeled_weights.append(torch.norm(torch.sub(prototypes[cid], labeled_prototypes[labeledid])))
                    labeled_weights = [w/sum(labeled_weights) for w in labeled_weights]
                    labeled_model = fmodule._model_sum([k * p for k, p in zip(labeled_models, labeled_weights)])
                elif self.agg_label_mode == 'proto_dis_max':
                    labeled_model = labeled_models[0]
                    max_dis = torch.norm(torch.sub(prototypes[cid], labeled_prototypes[0])).item()
                    for labeledid in range(1, len(labeled_models)):
                        dis = torch.norm(torch.sub(prototypes[cid], labeled_prototypes[labeledid])).item()
                        if max_dis < dis:
                            max_dis = dis
                            labeled_model = labeled_models[labeledid]
                    labeled_model = copy.deepcopy(labeled_model)
                # 'outs_sim', 'outs_sim_max', 'outs_dis', 'outs_dis_max'
                elif self.agg_label_mode == 'outs_sim':
                    labeled_weights = []
                    for labeledid in range(len(labeled_models)):
                        labeled_weights.append(F.cosine_similarity(outs[cid].view(self.num_normal, -1), labeled_outs[labeledid].view(self.num_normal, -1)).mean())
                    labeled_weights = [math.exp(labeled_weight) for labeled_weight in labeled_weights]
                    labeled_weights = [w/sum(labeled_weights) for w in labeled_weights]
                    labeled_model = fmodule._model_sum([k * p for k, p in zip(labeled_models, labeled_weights)])
                elif self.agg_label_mode == 'outs_sim_max':
                    labeled_model = labeled_models[0]
                    max_sim = F.cosine_similarity(outs[cid].view(self.num_normal, -1), labeled_outs[0].view(self.num_normal, -1)).mean().item()
                    for labeledid in range(1, len(labeled_models)):
                        sim = F.cosine_similarity(outs[cid].view(self.num_normal, -1), labeled_outs[labeledid].view(self.num_normal, -1)).mean().item()
                        if max_sim < sim:
                            max_sim = sim
                            labeled_model = labeled_models[labeledid]
                    labeled_model = copy.deepcopy(labeled_model)
                elif self.agg_label_mode == 'outs_dis':
                    labeled_weights = []
                    for labeledid in range(len(labeled_models)):
                        labeled_weights.append(torch.norm(torch.sub(outs[cid], labeled_outs[labeledid])))
                    labeled_weights = [w/sum(labeled_weights) for w in labeled_weights]
                    labeled_model = fmodule._model_sum([k * p for k, p in zip(labeled_models, labeled_weights)])
                elif self.agg_label_mode == 'outs_dis_max':
                    labeled_model = labeled_models[0]
                    max_dis = torch.norm(torch.sub(outs[cid], labeled_outs[0])).item()
                    for labeledid in range(1, len(labeled_models)):
                        dis = torch.norm(torch.sub(outs[cid], labeled_outs[labeledid])).item()
                        if max_dis < dis:
                            max_dis = dis
                            labeled_model = labeled_models[labeledid]
                    labeled_model = copy.deepcopy(labeled_model)
                # 'model_sim', 'model_sim_max', 'model_dis', 'model_dis_max'
                elif self.agg_label_mode == 'model_sim':
                    labeled_weights = []
                    for labeledid in range(len(labeled_models)):
                        labeled_weights.append(fmodule.cos_sim(models[cid], labeled_models[labeledid]))
                    labeled_weights = [math.exp(labeled_weight) for labeled_weight in labeled_weights]
                    labeled_weights = [w/sum(labeled_weights) for w in labeled_weights]
                    labeled_model = fmodule._model_sum([k * p for k, p in zip(labeled_models, labeled_weights)])
                elif self.agg_label_mode == 'model_sim_max':
                    labeled_model = labeled_models[0]
                    max_sim = fmodule.cos_sim(models[cid], labeled_models[0]).item()
                    for labeledid in range(1, len(labeled_models)):
                        sim = fmodule.cos_sim(models[cid], labeled_models[labeledid]).item()
                        if max_sim < sim:
                            max_sim = sim
                            labeled_model = labeled_models[labeledid]
                    labeled_model = copy.deepcopy(labeled_model)
                elif self.agg_label_mode == 'model_dis':
                    labeled_weights = []
                    for labeledid in range(len(labeled_models)):
                        dist_total = self.model_dis(models[cid], labeled_models[labeledid])
                        labeled_weights.append(dist_total)
                    labeled_weights = [w/sum(labeled_weights) for w in labeled_weights]
                    labeled_model = fmodule._model_sum([k * p for k, p in zip(labeled_models, labeled_weights)])
                elif self.agg_label_mode == 'model_dis_max':
                    labeled_model = labeled_models[0]
                    max_dis = self.model_dis(models[cid], labeled_models[0]).item()
                    for labeledid in range(1, len(labeled_models)):
                        dis = self.model_dis(models[cid], labeled_models[labeledid]).item()
                        if max_dis < dis:
                            max_dis = dis
                            labeled_model = labeled_models[labeledid]
                    labeled_model = copy.deepcopy(labeled_model)
                # 'random', 'weighted_scale'
                elif self.agg_label_mode == 'random':
                    labeled_weights = []
                    for labeledid in range(len(labeled_models)):
                        labeled_weights.append(random.random())
                    labeled_weights = [w/sum(labeled_weights) for w in labeled_weights]
                    labeled_model = fmodule._model_sum([k * p for k, p in zip(labeled_models, labeled_weights)])
                elif self.agg_label_mode == 'weighted_scale':
                    labeled_weights = [vol/sum(labeled_datavols) for vol in labeled_datavols]
                    labeled_model = fmodule._model_sum([k * p for k, p in zip(labeled_models, labeled_weights)])

                agg_labeled_models.append(labeled_model)
        return agg_labeled_models


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None, islabeled=True):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.islabeled = islabeled
        self.weight_contrastive = option['weight_contrastive']
        self.num_clusters = option['num_clusters']
        self.aggregate = option['aggregate']
        self.contrastive = option['contrastive']
        self.size_bank = option['size_bank']
        self.ema = option['ema']
        self.prototype_bank = []

    @fmodule.with_multi_gpus
    def train(self, model, labeled_model, image_prototype):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        """
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.train_one_step(model, batch_data, labeled_model, image_prototype, self.contrastive, self.weight_contrastive, self.islabeled)['loss']
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
        return

    @fmodule.with_multi_gpus
    def get_entropy(self, model):
        return self.calculator.get_entropy(model, self.train_data)

    @fmodule.with_multi_gpus
    def get_consistency(self, model, labeled_model):
        return self.calculator.get_consistency(model, labeled_model, self.train_data)

    @fmodule.with_multi_gpus
    def get_prototypes(self, model, labeled_model):
        fore_prototypes, back_prototypes = self.calculator.get_prototypes(model, labeled_model, self.train_data)

        image_prototypes = [None for _ in range(2)]

        if len(fore_prototypes) == 0:
            client_prototype = None
        else:
            fore_prototypes = torch.stack(fore_prototypes)
            client_prototype = fore_prototypes.mean(0)

        if self.islabeled:
            if not isinstance(fore_prototypes, list):
                fore_prototypes = np.array(fore_prototypes.cpu())
                if fore_prototypes.shape[0] < self.num_clusters or self.num_clusters == -1:
                    image_prototypes[0] = fore_prototypes
                else:
                    results = k_means(fore_prototypes, n_clusters=self.num_clusters)
                    image_prototypes[0] = results[0]
                image_prototypes[0] = torch.Tensor(image_prototypes[0]).to(self.device)
            if len(back_prototypes) != 0:
                back_prototypes = np.array(torch.stack(back_prototypes).cpu())
                if back_prototypes.shape[0] < self.num_clusters or self.num_clusters == -1:
                    image_prototypes[1] = back_prototypes
                else:
                    results = k_means(back_prototypes, n_clusters=self.num_clusters)
                    image_prototypes[1] = results[0]
                image_prototypes[1] = torch.Tensor(image_prototypes[1]).to(self.device)

        return client_prototype, image_prototypes

    def push_prototype(self, image_prototype):
        if image_prototype is None and len(self.prototype_bank) == 0:
            return None

        if len(self.prototype_bank) >= self.size_bank:
            self.prototype_bank.pop(0)

        temp_prototypes = [[], []]
        if len(self.prototype_bank) != 0:
            for ips in self.prototype_bank:
                temp_prototypes[0].append(ips[0])
                temp_prototypes[1].append(ips[1])
            temp_prototypes[0] = torch.cat(temp_prototypes[0], dim=0)
            temp_prototypes[1] = torch.cat(temp_prototypes[1], dim=0)

        if image_prototype is not None:
            self.prototype_bank.append(image_prototype)
            if isinstance(temp_prototypes[0], list):
                temp_prototypes[0] = image_prototype[0]
                temp_prototypes[1] = image_prototype[1]
            else:
                fp = image_prototype[0].repeat(len(self.prototype_bank)-1, 1)
                size_fp = min(fp.shape[0], temp_prototypes[0].shape[0])
                fp = fp[:size_fp, :]
                temp_prototypes[0] = temp_prototypes[0][:size_fp, :]
                bp = image_prototype[1].repeat(len(self.prototype_bank)-1, 1)
                size_bp = min(bp.shape[0], temp_prototypes[1].shape[0])
                bp = bp[:size_bp, :]
                temp_prototypes[1] = temp_prototypes[1][:size_bp, :]

                temp_prototypes[0] = self.ema * temp_prototypes[0] + (1-self.ema) * fp
                temp_prototypes[1] = self.ema * temp_prototypes[1] + (1-self.ema) * bp
                temp_prototypes[0] = torch.cat([temp_prototypes[0], image_prototype[0]], dim=0)
                temp_prototypes[1] = torch.cat([temp_prototypes[1], image_prototype[1]], dim=0)

        return temp_prototypes

    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return list(received_pkg.values())

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the updated
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model, labeled_model, image_prototype = self.unpack(svr_pkg)

        image_prototype = self.push_prototype(image_prototype)

        if self.aggregate == 'weighted_entropy_dis':
            entropy1 = self.get_entropy(model)
        self.train(model, labeled_model, image_prototype)
        if self.aggregate.startswith('weighted_entropy'):
            entropy2 = self.get_entropy(model)

        agg_statistics = None
        if self.aggregate == 'weighted_entropy':
            agg_statistics = entropy2
        elif self.aggregate == 'weighted_entropy_dis':
            entropy_dis = entropy1 - entropy2
            agg_statistics = entropy_dis
        elif self.aggregate == 'weighted_consistency':
            consistency = self.get_consistency(model, labeled_model)
            print(consistency)
            agg_statistics = consistency

        client_prototype, image_prototype = self.get_prototypes(model, labeled_model)
        cpkg = self.pack(model, client_prototype, image_prototype, agg_statistics)
        return cpkg

    def pack(self, model, client_prototype, image_prototype, agg_statistics):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        """
        return {
            "model": model,
            "client_prototype": client_prototype,
            "image_prototype": image_prototype,
            "agg_statistics": agg_statistics
        }
