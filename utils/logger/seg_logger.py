import utils.logger.basic_logger as bl
import numpy as np
import random


class Logger(bl.Logger):

    def __init__(self, meta, labeledclients, *args, **kwargs):
        super(Logger, self).__init__(meta, *args, **kwargs)
        self.labeledclients = labeledclients

    def get_output_name(self, suffix='.json'):
        if not hasattr(self, 'meta'):
            raise NotImplementedError('logger has no attr named "meta"')
        header = "{}_".format(self.meta["algorithm"])
        if hasattr(self, 'server'):
            for para, pv in self.server.algo_para.items():
                header = header + para + "{}_".format(pv)
        else:
            if self.meta['algo_para'] is not None:
                header = header + 'algopara_' + '|'.join([str(p) for p in self.meta['algo_para']])

        output_name = header + 'LB' + ''.join([str(lc) for lc in self.labeledclients]) + '_'

        output_name = output_name + "M{}_R{}_B{}_".format(self.meta['model'], self.meta['num_rounds'], self.meta['batch_size'])
        if self.meta['num_steps'] < 0:
            output_name = output_name + ("E{}_".format(self.meta['num_epochs']))
        else:
            output_name = output_name + ("K{}_".format(self.meta['num_steps']))

        output_name = output_name + "LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_NET{}_CMP{}_".format(
            self.meta['learning_rate'],
            self.meta['proportion'],
            self.meta['seed'],
            self.meta['lr_scheduler'] + self.meta['learning_rate_decay'],
            self.meta['weight_decay'],
            self.meta['network_config'],
            self.meta['computing_config']
        )

        output_name = output_name + 'NO' + self.meta['sample'] + '+' + self.meta['aggregate'] + '+' + self.meta['aggregate_labeled'] + '+' + str(self.meta['contrastive'])

        output_name = output_name + suffix
        return output_name

    def log_per_round(self, *args, **kwargs):
        """This method is called at the beginning of each communication round of Server.
        The round-wise operations of recording should be complemented here."""
        # calculate the testing metrics on testing dataset owned by server
        test_metric = self.server.test()
        if 'Xray' not in self.meta['task']:
            for met_name, met_val in test_metric.items():
                self.output['test_' + met_name + '_dist'].append(met_val)
                self.output['test_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(
                    self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
        else:
            for met_name, met_val in test_metric.items():
                self.output['test_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across clients
        train_metrics = self.server.test_on_clients('train')
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name + '_dist'].append(met_val)
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(
                self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
        # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        valid_metrics = self.server.test_on_clients('valid')
        for met_name, met_val in valid_metrics.items():
            self.output['valid_'+met_name+'_dist'].append(met_val)
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(
                self.server.local_data_vols, met_val)]) / self.server.total_data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()
