class PatternMatcher(object):
    def __init__(self, predictor):
        self.predictor = predictor


    def train(self, ws, ps, ls):
        if not ps:
            ps = [None] * len(ws)

        for w, p, l in zip(ws, ps, ls):
            if not p:
                p = [None] * len(w)

            for wi, pi, li in zip(w, p, l):
                self.predictor.update(wi, pi, li)


    def decode(self, ws, ps):
        ys = []

        if not ps:
            ps = [None] * len(ws)

        for w, p in zip(ws, ps):
            if not p:
                p = [None] * len(w)

            y = []
            for wi, pi in zip(w, p):
                yi = self.predictor.predict(wi, pi)
                if yi: # [(word1, freq1)]
                    yi = yi[0][0] 

                y.append(yi)

            ys.append(y)

        return ys


    def decode_k_best(self, ws, ps, k=1):
        ys = []

        if not ps:
            ps = [None] * len(ws)

        for w, p in zip(ws, ps):
            if not p:
                p = [None] * len(w)

            y = []
            for wi, pi in zip(w, p):
                yi = self.predictor.predict_k_best(wi, pi, k)
                if yi: # [(word1, freq1), (word2, freq2), ...]
                    yi = [yi[j][0] for j in range(len(yi))] 

                y.append(yi)

            ys.append(y)

        return ys
