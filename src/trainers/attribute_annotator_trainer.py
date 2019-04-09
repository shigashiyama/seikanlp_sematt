import copy
from datetime import datetime
import pickle
import sys

import classifiers.pattern_matcher
import common
import constants, constants_sematt
from data_loaders import data_loader, attribute_annotation_data_loader
from evaluators.common import AccuracyEvaluator
import models.attribute_annotator
from trainers import trainer
from trainers.trainer import Trainer
import util


class SimpleTrainer(object):
    def __init__(self, args, logger=sys.stderr):
        err_msgs = []
        if args.execute_mode == 'train':
            if not args.train_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--train_data')
                err_msgs.append(msg)
            if not args.task and not args.model_path:
                msg = 'Error: the following argument is required for {} mode unless resuming a trained model: {}'.format('train', '--task')
                err_msgs.append(msg)

        elif args.execute_mode == 'eval':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
                err_msgs.append(msg)

            if not args.test_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--test_data')
                err_msgs.append(msg)

        elif args.execute_mode == 'decode':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
                err_msgs.append(msg)

            if not args.decode_data:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--decode_data')
                err_msgs.append(msg)

        elif args.execute_mode == 'interactive':
            if not args.model_path:
                msg = 'Error: the following argument is required for {} mode: {}'.format(
                    args.execute_mode, '--model_path/-m')
                err_msgs.append(msg)

        else:
            msg = 'Error: invalid execute mode: {}'.format(args.execute_mode)
            err_msgs.append(msg)

        if err_msgs:
            for msg in err_msgs:
                print(msg, file=sys.stderr)
            sys.exit()

        self.args = args
        self.start_time = datetime.now().strftime('%Y%m%d_%H%M')
        self.logger = logger    # output execute log
        self.reporter = None    # output evaluation results
        self.task = args.task
        self.train = None
        self.dev = None
        self.test = None
        self.decode_data = None
        self.hparams = None
        self.dic = None
        self.dic_org = None
        self.dic_dev = None
        self.dic_ext = None # tmp
        self.data_loader = None
        self.classifier = None
        self.evaluator = None
        # self.feat_extractor = None
        # self.optimizer = None
        # self.n_iter = 1
        self.label_begin_index = 2

        self.log('Start time: {}\n'.format(self.start_time))
        if not self.args.quiet:
            self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), 'a')


    def report(self, message):
        if not self.args.quiet:
            print(message, file=self.reporter)


    def log(self, message=''):
        print(message, file=self.logger)


    def close(self):
        if not self.args.quiet:
            self.reporter.close()


    def load_external_embedding_models(self):
        pass


    def load_external_dictionary(self):
        pass


    def init_feature_extractor(self, use_gpu):
        pass


    def load_training_and_development_data(self):
        self.load_data('train')
        if self.args.devel_data:
            self.load_data('devel')
        self.show_training_data()


    def load_test_data(self):
        self.dic_org = copy.deepcopy(self.dic)
        self.load_data('test')


    def load_decode_data(self):
        self.load_data('decode')


    def load_data(self, data_type):
        if data_type == 'train':
            self.setup_data_loader()
            data_path = self.args.path_prefix + self.args.train_data
            data, self.dic = self.data_loader.load_gold_data(
                data_path, self.args.input_data_format, dic=self.dic, train=True)
            self.dic.create_id2strs()
            self.train = data

        elif data_type == 'devel':
            data_path = self.args.path_prefix + self.args.devel_data
            self.dic_dev = copy.deepcopy(self.dic)
            data, self.dic_dev = self.data_loader.load_gold_data(
                data_path, self.args.input_data_format, dic=self.dic_dev, train=False)
            self.dic_dev.create_id2strs()
            self.dev = data

        elif data_type == 'test':
            self.setup_data_loader()
            data_path = self.args.path_prefix + self.args.test_data
            data, self.dic = self.data_loader.load_gold_data(
                data_path, self.args.input_data_format, dic=self.dic, train=False)
            self.dic.create_id2strs()
            self.test = data

        elif data_type == 'decode':
            self.setup_data_loader()
            data_path = self.args.path_prefix + self.args.decode_data
            data = self.data_loader.load_decode_data(
                data_path, self.args.input_data_format, dic=self.dic)
            self.dic.create_id2strs()
            self.decode_data = data
        
        else:
            print('Error: incorect data type: {}'.format(), file=sys.stderr)
            sys.exit()

        self.log('Load {} data: {}'.format(data_type, data_path))
        self.show_data_info(data_type)


    def init_model(self):
        self.log('Initialize model from hyperparameters\n')
        self.setup_classifier()


    def load_dic(self, dic_path):
        with open(dic_path, 'rb') as f:
            self.dic = pickle.load(f)
        self.log('Load dic: {}'.format(dic_path))
        self.log('Num of tokens: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        if self.dic.has_table(constants.BIGRAM):
            self.log('Num of bigrams: {}'.format(len(self.dic.tables[constants.BIGRAM])))
        if self.dic.has_trie(constants.CHUNK):
            self.log('Num of chunks: {}'.format(len(self.dic.tries[constants.CHUNK])))
        if self.dic.has_table(constants.SEG_LABEL):
            self.log('Num of segmentation labels: {}'.format(len(self.dic.tables[constants.SEG_LABEL])))
        for i in range(3):      # tmp
            if self.dic.has_table(constants.ATTR_LABEL(i)):
                self.log('Num of {}-th attribute labels: {}'.format(
                    i, len(self.dic.tables[constants.ATTR_LABEL(i)])))
        if self.dic.has_table(constants.ARC_LABEL):
            self.log('Num of arc labels: {}'.format(len(self.dic.tables[constants.ARC_LABEL])))
        if self.dic.has_table(constants_sematt.SEM_LABEL):
            self.log('Num of sem labels: {}'.format(len(self.dic.tables[constants_sematt.SEM_LABEL])))
        self.log('')


    def show_hyperparameters(self):
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            if (k in self.hparams and
                ('dropout' in k or 'freq_threshold' in k or 'max_vocab_size' in k or
                 k == 'attr_indexes' )):
                update = self.hparams[k] != v
                message = '{}={}{}'.format(
                    k, v, ' (original value ' + str(self.hparams[k]) + ' was updated)' if update else '')
                self.hparams[k] = v

            elif (k == 'task' and v == 'seg' and self.hparams[k] == constants.TASK_SEGTAG and 
                  (self.args.execute_mode == 'decode' or self.args.execute_mode == 'interactive')):
                self.task = self.hparams[k] = v
                message = '{}={}'.format(k, v)            

            elif k in self.hparams and v != self.hparams[k]:
                message = '{}={} (input option value {} was discarded)'.format(k, self.hparams[k], v)

            else:
                message = '{}={}'.format(k, v)

            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def update_model(self, classifier=None, dic=None, train=False):
        pass


    def run_eval_mode(self):
        self.log('<test result>')
        res = self.run_epoch(self.test, train=False)
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('test\t%s\n' % res)
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_decode_mode(self):
        self.update_model(classifier=self.classifier, dic=self.dic) 
        if self.args.output_data:
            with open(self.args.output_data, 'w') as f:
                self.decode(self.decode_data, file=f)
        else:
            self.decode(self.decode_data, file=sys.stdout)


class AttributeAnnotatorTrainer(SimpleTrainer):
    def __init__(self, args, logger=sys.stderr):
        super().__init__(args, logger)
        self.label_begin_index = 2


    def show_data_info(self, data_type):
        dic = self.dic_dev if data_type == 'devel' else self.dic
        self.log('### {} dic'.format(data_type))
        self.log('Num of tokens: {}'.format(len(dic.tables[constants.UNIGRAM])))
        

    def show_training_data(self):
        train = self.train
        dev = self.dev
        self.log('### Loaded data')
        self.log('# train: {} ... {}\n'.format(train.inputs[0][0], train.inputs[0][-1]))
        self.log('# train_gold_attr: {} ... {}\n'.format(train.outputs[0][0], train.outputs[0][-1]))
        t2i_tmp = list(self.dic.tables[constants.UNIGRAM].str2id.items())
        self.log('# token2id: {} ... {}\n'.format(t2i_tmp[:10], t2i_tmp[len(t2i_tmp)-10:]))

        attr_indexes=common.get_attribute_values(self.args.attr_indexes)
        for i in range(len(attr_indexes)):
            if self.dic.has_table(constants.ATTR_LABEL(i)):
                id2attr = {v:k for k,v in self.dic.tables[constants.ATTR_LABEL(i)].str2id.items()}
                self.log('# {}-th attribute labels: {}\n'.format(i, id2attr))
        
        self.report('[INFO] vocab: {}'.format(len(self.dic.tables[constants.UNIGRAM])))
        self.report('[INFO] data length: train={} devel={}'.format(
            len(train.inputs[0]), len(dev.inputs[0]) if dev else 0))


    def init_hyperparameters(self):
        self.hparams = {
            'task' : self.args.task,
            'lowercasing' : self.args.lowercasing,
            'normalize_digits' : self.args.normalize_digits,
        }

        self.log('Init hyperparameters')
        self.log('### arguments')
        for k, v in self.args.__dict__.items():
            message = '{}={}'.format(k, v)
            self.log('# {}'.format(message))
            self.report('[INFO] arg: {}'.format(message))
        self.log('')


    def load_model(self):
        model_path = self.args.model_path
        if model_path.endswith('.pkl'):
            model_format = 'pkl'
            array = model_path.split('.pkl')
        elif model_path.endswith('.txt'):
            model_format = 'txt'
            array = model_path.split('.txt')
        else:
            print('Error: invalid model format. The file name must ends with \'pkl\' or \'txt\'.', 
                  file=sys.stderr)
            sys.exit()

        dic_path = '{}.s2i'.format(array[0])
        hparam_path = '{}.hyp'.format(array[0])
        param_path = model_path

        # dictionary
        self.load_dic(dic_path)

        # hyper parameters
        self.load_hyperparameters(hparam_path)
        self.log('Load hyperparameters: {}\n'.format(hparam_path))
        self.show_hyperparameters()

        # model
        if model_format == 'pkl':
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
                
        elif model_format == 'txt':
            predictor = models.attribute_annotator.load_model_from_txt(
                model_path,
                self.dic.tables[constants_sematt.SEM_LABEL].str2id,
                self.dic.tables[constants.UNIGRAM].str2id,
                (self.dic.tables[constants.ATTR_LABEL(0)].str2id
                if self.dic.has_table(constants.ATTR_LABEL(0)) else None))
            self.classifier = classifiers.pattern_matcher.PatternMatcher(predictor)

        self.log('Load model: {}\n'.format(model_path))


    def load_hyperparameters(self, hparams_path):
        hparams = {}
        with open(hparams_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue

                kv = line.split('=')
                key = kv[0]
                val = kv[1]

                if (key == 'lowercasing' or
                    key == 'normalize_digits'
                ):
                    val = (val.lower() == 'true')

                hparams[key] = val

        self.hparams = hparams
        self.task = self.hparams['task']


    def setup_data_loader(self):
        attr_indexes=common.get_attribute_values(self.args.attr_indexes)
        self.data_loader = attribute_annotation_data_loader.AttributeAnnotationDataLoader(
            token_index=self.args.token_index,
            label_index=self.args.label_index,
            attr_indexes=attr_indexes,
            attr_depths=common.get_attribute_values(self.args.attr_depths, len(attr_indexes)),
            attr_chunking_flags=common.get_attribute_boolvalues(
                self.args.attr_chunking_flags, len(attr_indexes)),
            attr_target_labelsets=common.get_attribute_labelsets(
                self.args.attr_target_labelsets, len(attr_indexes)),
            attr_delim=self.args.attr_delim,
            lowercasing=self.hparams['lowercasing'],
            normalize_digits=self.hparams['normalize_digits'],
        )


    def setup_classifier(self):
        predictor = models.attribute_annotator.AttributeAnnotator()
        self.classifier = classifiers.pattern_matcher.PatternMatcher(predictor)


    def setup_optimizer(self):
        pass


    def setup_evaluator(self, evaluator=None):
        ignored_labels = set()
        if self.args.ignored_labels:
            for label in self.args.ignored_labels.split(','):
                label_id = self.dic.tables[constants.ATTR_LABEL(0)].get_id(label)
                if label_id >= 0:
                    ignored_labels.add(label_id)
            self.args.ignored_labels = ignored_labels
                
        self.log('Setup evaluator: labels to be ignored={}\n'.format(ignored_labels))
        self.evaluator = AccuracyEvaluator(
            ignore_head=False, ignored_labels=ignored_labels)


    def gen_inputs(self, data, evaluate=True):
        xs = data.inputs[0]
        ps = data.inputs[0] if data.inputs[0] else None
        ls = data.outputs[0] if evaluate else None
        if evaluate:
            return xs, ps, ls
        else:
            return xs, ps


    def decode(self, rdata, file=sys.stdout):
        org_tokens = rdata.orgdata[0]
        org_attrs = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None
        n_ins = len(org_tokens)

        timer = util.Timer()
        timer.start()
        inputs = self.gen_inputs(rdata, evaluate=False)
        self.decode_batch(*inputs, org_tokens=org_tokens, org_attrs=org_attrs, file=file)
        timer.stop()

        print('Parsed %d sentences. Elapsed time: %.4f sec (total) / %.4f sec (per sentence)' % (
            n_ins, timer.elapsed, timer.elapsed/n_ins), file=sys.stderr)


    def decode_batch(self, *inputs, org_tokens=None, org_attrs=None, file=sys.stdout):
        id2label = self.dic.tables[constants_sematt.SEM_LABEL].id2str
        ls = self.classifier.decode_k_best(*inputs, k=self.args.k_best)

        use_attr = org_attrs is not None
        if not use_attr:
            org_attrs = [None] * len(org_tokens)

        for x_str, p_str, l in zip(org_tokens, org_attrs, ls):
            l_str = [','.join([id2label[int(j)] for j in li]) for li in l]

            if use_attr:
                res = ['{}\t{}\t{}'.format(
                    xi_str, pi_str, li_str) for xi_str, pi_str, li_str in zip(x_str, p_str, l_str)]
            else:
                res = ['{}\t{}'.format(xi_str, li_str) for xi_str, li_str in zip(x_str, l_str)]
            res = '\n'.join(res).rstrip()

            print(res+'\n', file=file)
            # print(res, file=file)


    def run_epoch(self, data, train=True):
        classifier = self.classifier
        evaluator = self.evaluator

        inputs = self.gen_inputs(data)
        xs = inputs[0]
        n_sen = len(xs)

        golds = inputs[self.label_begin_index]
        if train:
            self.classifier.train(*inputs)
        ret = self.classifier.decode(*inputs[:self.label_begin_index])
        counts = self.evaluator.calculate(*[xs], *[golds], *[ret])

        if train:
            self.log('\n<training result>')
            res = evaluator.report_results(n_sen, counts, file=self.logger)
            self.report('train\t%s' % res)

            if self.args.devel_data:
                self.log('\n<development result>')
                v_res = self.run_epoch(self.dev, train=False)
                self.report('devel\t%s' % v_res)

            # save model
            if not self.args.quiet:
                mdl_path = '{}/{}.pkl'.format(constants.MODEL_DIR, self.start_time)
                with open(mdl_path, 'wb') as f:
                    pickle.dump(self.classifier, f)

                mdl_path_txt = '{}/{}.txt'.format(constants.MODEL_DIR, self.start_time)
                self.classifier.predictor.dump_model_as_txt(
                    mdl_path_txt,
                    self.dic.tables[constants_sematt.SEM_LABEL].id2str,
                    self.dic.tables[constants.UNIGRAM].id2str,
                    self.dic.tables[constants.ATTR_LABEL(0)].id2str)
                    
                self.log('Save the model (binary): %s' % mdl_path)
                self.log('Save the model (text): %s' % mdl_path_txt)
                self.report('[INFO] Save the model (binary): %s\n' % mdl_path)
                self.report('[INFO] Save the model (text1): %s\n' % mdl_path_txt)

            if not self.args.quiet:
                self.reporter.close() 
                self.reporter = open('{}/{}.log'.format(constants.LOG_DIR, self.start_time), 'a')

        res = None if train else evaluator.report_results(n_sen, counts, file=self.logger)
        return res


    def run_train_mode(self):
        # save model
        if not self.args.quiet:
            hparam_path = '{}/{}.hyp'.format(constants.MODEL_DIR, self.start_time)
            with open(hparam_path, 'w') as f:
                for key, val in self.hparams.items():
                    print('{}={}'.format(key, val), file=f)
                self.log('Save hyperparameters: {}'.format(hparam_path))

            dic_path = '{}/{}.s2i'.format(constants.MODEL_DIR, self.start_time)
            with open(dic_path, 'wb') as f:
                pickle.dump(self.dic, f)
            self.log('Save string2index table: {}'.format(dic_path))

        # training
        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.log('Start epoch: {}\n'.format(time))
        self.report('[INFO] Start epoch at {}'.format(time))

        self.run_epoch(self.train, train=True)

        time = datetime.now().strftime('%Y%m%d_%H%M')
        self.report('Finish: %s\n' % time)
        self.log('Finish: %s\n' % time)


    def run_interactive_mode(self):
        print('Please input text or type \'q\' to quit this mode:')
        while True:
            line = sys.stdin.readline().rstrip(' \t\n')
            if len(line) == 0:
                continue
            elif line == 'q':
                break

            rdata = self.data_loader.parse_commandline_input(line, self.dic)
            inputs = self.gen_inputs(rdata, evaluate=False)
            ot = rdata.orgdata[0]
            oa = rdata.orgdata[1] if len(rdata.orgdata) > 1 else None

            self.decode_batch(*inputs, org_tokens=ot, org_attrs=oa)
