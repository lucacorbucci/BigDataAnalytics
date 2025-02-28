import numpy as np


class Explanation(object):

    def __init__(self):
        self.bb_pred = None
        self.dt_pred = None

        self.rule = None
        self.crules = None
        self.deltas = None

        self.fidelity = None
        self.dt = None

    def __str__(self):
        deltas_str = '{ '
        for i, delta in enumerate(self.deltas):
            deltas_str += '      { ' if i > 0 else '{ '
            deltas_str += ', '.join([str(s) for s in delta])
            deltas_str += ' },\n'
        deltas_str = deltas_str[:-2] + ' }'
        return 'r = %s\nc = %s' % (self.rule, deltas_str)

    def rstr(self):
        return self.rule

    def cstr(self):
        deltas_str = '{ '
        for i, delta in enumerate(self.deltas):
            deltas_str += '{ ' if i > 0 else '{ '
            deltas_str += ', '.join([str(s) for s in delta])
            deltas_str += ' } --> %s, ' % self.crules[i]._cstr()
        deltas_str = deltas_str[:-2] + ' }'
        return deltas_str


class MultilabelExplanation(Explanation):
    def __init__(self):
        super(MultilabelExplanation).__init__()
        self.dt_list = None
        self.rule_list = None
        self.crules_list = None
        self.deltas_list = None


class ImageExplanation(Explanation):
    def __init__(self, img, segments):
        super(ImageExplanation).__init__()
        self.img = img
        self.segments = segments

    def get_image_rule(self, hide_rest=False, num_features=None, min_importance=0.0):
        mask = np.zeros(self.segments.shape, self.segments.dtype)

        if hide_rest:
            img2show = np.zeros(self.img.shape).astype(int)
        else:
            img2show = np.copy(self.img)

        num_features = len(self.dt.feature_importances_) if num_features is None else num_features
        features = np.argsort(self.dt.feature_importances_)[:num_features]

        for p in self.rule.premises:
            if p.att not in features or self.dt.feature_importances_[p.att] < min_importance:
                continue
            f = p.att
            w = -1 if p.op == '<=' else 1

            c = 0 if w < 0 else 1
            mask[self.segments == f] = 1 if w < 0 else 2
            img2show[self.segments == f] = self.img[self.segments == f].copy()
            if not hide_rest:
                img2show[self.segments == f, c] = np.max(self.img)
            for cp in [0, 1, 2]:
                if c == cp:
                    continue
        return img2show, mask

    def get_image_counterfactuals(self, hide_rest=False, num_features=None, min_importance=0.0):
        imgs2show, masks = list(), list()
        coutcomes = list()
        for delta, crule in zip(self.deltas, self.crules):
            mask = np.zeros(self.segments.shape, self.segments.dtype)

            if hide_rest:
                img2show = np.zeros(self.img.shape).astype(int)
            else:
                img2show = np.copy(self.img)

            num_features = len(self.dt.feature_importances_) if num_features is None else num_features
            features = np.argsort(self.dt.feature_importances_)[:num_features]

            for p in delta:
                if p.att not in features or self.dt.feature_importances_[p.att] < min_importance:
                    continue
                f = p.att
                w = -1 if p.op == '<=' else 1

                c = 0 if w < 0 else 1
                mask[self.segments == f] = 1 if w < 0 else 2
                img2show[self.segments == f] = self.img[self.segments == f].copy()
                if not hide_rest:
                    img2show[self.segments == f, c] = np.max(self.img)
                for cp in [0, 1, 2]:
                    if c == cp:
                        continue

                imgs2show.append(img2show)
                masks.append(mask)
                coutcomes.append(crule.cons)

        return imgs2show, masks, coutcomes


class TextExplanation(Explanation):
    def __init__(self, text, indexed_text):
        super(TextExplanation).__init__()
        self.text = text
        self.indexed_text = indexed_text

    def get_text_rule(self, num_features=None, min_importance=0.0):

        num_features = len(self.dt.feature_importances_) if num_features is None else num_features
        features = np.argsort(self.dt.feature_importances_)[:num_features]

        inwords, outwords = list(), list()
        for p in self.rule.premises:
            if p.att not in features or self.dt.feature_importances_[p.att] < min_importance:
                continue

            word = self.indexed_text.word(p.att)
            if p.op == '<=':
                outwords.append(word)
            else:
                inwords.append(word)

        text_premise = ', '.join(inwords) if len(inwords) > 0 else ''
        text_premise += ', ' if len(inwords) > 0 and len(outwords) > 0 else ''
        text_premise += ', '.join(['¬ %s' % word for word in outwords]) if len(outwords) > 0 else ''
        text_rule = '{ %s } --> %s' % (text_premise, self.rule._cstr())
        return text_rule

    def get_text_counterfactuals(self, num_features=None, min_importance=0.0):

        num_features = len(self.dt.feature_importances_) if num_features is None else num_features
        features = np.argsort(self.dt.feature_importances_)[:num_features]

        text_counterfactuals = list()
        for delta, crule in zip(self.deltas, self.crules):

            inwords, outwords = list(), list()
            for p in delta:
                if p.att not in features or self.dt.feature_importances_[p.att] < min_importance:
                    continue

                word = self.indexed_text.word(p.att)
                if p.op == '<=':
                    outwords.append(word)
                else:
                    inwords.append(word)

            text_premise = ', '.join(inwords) if len(inwords) > 0 else ''
            text_premise += ', ' if len(inwords) > 0 and len(outwords) > 0 else ''
            text_premise += ', '.join(['¬ %s' % word for word in outwords]) if len(outwords) > 0 else ''
            text_rule = '{ %s } --> %s' % (text_premise, crule._cstr())
            text_counterfactuals.append(text_rule)

        return text_counterfactuals
