import collections


class Key2Counter(object):
    def __init__(self):
        self.key2counter = {}
        
        
    def __len__(self):
        return len(self.key2counter)
        
        
    def __str__(self):
        return str(self.key2counter)
        
        
    def increment(self, key, val):
        if not key in self.key2counter:
            self.key2counter[key] = collections.Counter()
        self.key2counter[key][val] += 1


    def add(self, key, val, num):
        if not key in self.key2counter:
            self.key2counter[key] = collections.Counter()
        self.key2counter[key][val] += num


    def get(self, key):
        if key in self.key2counter:
            return self.key2counter[key]
        else:
            return None
            
            
    def keys(self):
        return self.key2counter.keys()


    # def write(self, path, key2str, val2str):
    #     with open(path, 'w') as f:
    #         f.write('#Key\tLabel1:Freq1, Label2:Freq2, ...\n')
    #         for key, counter in self.key2counter.items():
    #             ret = ', '.join(['{}:{}'.format(val2str[item[0]],item[1]) for item in counter.items()])
    #             if key in key2str:
    #                 f.write('{}\t{}\n'.format(key2str[key], ret))

    
class AttributeAnnotator(object):
    def __init__(self):
        self.word_dic = Key2Counter()
        self.pos_dic = Key2Counter()
        self.label_counter = collections.Counter()


    def update(self, word, pos, label):
        self.word_dic.increment(word, label)
        self.label_counter[label] += 1
        if pos:
            self.pos_dic.increment(pos, label)


    def predict(self, word, pos=None):
        return self.predict_k_best(word, pos, 1)


    def predict_k_best(self, word, pos=None, k=1):
        most_common_label = None

        if word in self.word_dic.key2counter:
            counter = self.word_dic.get(word)
            pred = counter.most_common(k)

        else:
            if pos in self.pos_dic.key2counter:
                counter = self.pos_dic.get(pos)
                pred = counter.most_common(k)

            else:
                if not most_common_label:
                    most_common_label = self.label_counter.most_common(k)
                pred = most_common_label

        return pred


    def dump_model_as_txt(self, path, label2str, word2str, pos2str):
        with open(path, 'w') as f:
            f.write('#Key\tLabel1:Freq1, ...\n')

            # pos_dic
            f.write('#<POS>\n')
            for key, counter in self.pos_dic.key2counter.items():
                ret = ', '.join(['{}:{}'.format(label2str[item[0]],item[1]) for item in counter.items()])
                if key in pos2str:
                    f.write('{}\t{}\n'.format(pos2str[key], ret))

            # word_dic
            f.write('#<WORD>\n')
            for key, counter in self.word_dic.key2counter.items():
                ret = ', '.join(['{}:{}'.format(label2str[item[0]],item[1]) for item in counter.items()])
                if key in word2str:
                    f.write('{}\t{}\n'.format(word2str[key], ret))
            

def load_model_from_txt(path, label2id, word2id, pos2id):
    model = AttributeAnnotator()

    is_pos = False
    is_word = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if not line: 
                continue

            elif line[0] == '#':
                if line == '#<POS>':
                    is_pos = True and pos2id is not None
                    is_word = False

                elif line == '#<WORD>':
                    is_pos = False
                    is_word = True

                continue

            key, values_str = line.split('\t')
            values = values_str.split(',')
            for val in values:
                label, freq = val.strip(' ').split(':')
                freq = int(freq)

                if is_word:
                    model.word_dic.add(word2id[key], label2id[label], freq)
                    model.label_counter[label] += freq
                elif is_pos:
                    model.pos_dic.add(pos2id[key], label2id[label], freq)
    
    return model
