import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    if len(feature_vector) < 2:
        return [], [], None, None
    n = len(feature_vector)
    data = np.c_[feature_vector, target_vector]
    data = data[data[:,0].argsort(kind='mergesort')]

    thresholds = np.empty(0)
    feature_vector_u = np.unique(feature_vector)
    thresholds = (feature_vector_u[:-1] + feature_vector_u[1:]) / 2

    if len(thresholds) == 0:
        return [], [], None, None

    n = len(data)
    l = []
    r = []
    ginis = []

    feature_left = data[:, 0]
    feature_left = np.append(feature_left, feature_left[-1])
    feature = np.array(feature_left)


    target = data[:, 1]
    sequence = np.cumsum(np.ones(len(target))) #[1, 2, 3, 4, 5, ..., n]
    left_1 = np.cumsum(target) # [1, 2, 3, ..., n_1]
    left_0 = sequence - left_1  #[1, 2, 3, ..., n_0]
    right_1 = np.sum(target) - left_1
    right_0 = (n - np.sum(target)) - left_0
    left = left_1 + left_0
    right = right_1 + right_0

    l_coef = left / n
    r_coef = right / n

    l_p1 = left_1 / left
    l_p0 = left_0 / left
    l_hr = 1 - (l_p1 ** 2) - (l_p0 ** 2)

    r_p1 = right_1 / right
    r_p0 = right_0 / right
    r_hr = 1 - (r_p1 ** 2) - (r_p0 ** 2)

    ginis = -l_coef * l_hr - r_coef * r_hr

    ginis = ginis[(feature[:-1] - feature[1:]) != 0]
    gini_best = np.max(ginis)
    threshold_best = thresholds[ginis.argmax()]


#    for i in range(len(thresholds)):
#        for j in range(n):
#            if data[j][0] < thresholds[i]:
#                l.append(data[j])
#            else:
#        l_coef = len(l) / len(data)
#        r_coef = len(r) / len(data)
#        count_0 = 0
#        count_1 = 0
#        for j in range(len(l)):
#            if l[j][1] == 1:
#                count_1 += 1
#            else:
#                count_0 += 1
#        l_p1 = count_1 / len(l)
#        l_p0 = count_0 / len(l)
#        l_hr = 1 - (l_p1 ** 2) - (l_p0 ** 2)
#        count_0 = 0
#        count_1 = 0
#        for j in range(len(r)):
#            if r[j][1] == 1:
#                count_1 += 1
#            else:
#                count_0 += 1
#        r_p1 = count_1 / len(r)
#        r_p0 = count_0 / len(r)
#        r_hr = 1 - (r_p1 ** 2) - (r_p0 ** 2)
#        qr = -l_coef * l_hr - r_coef * r_hr
#        l, r = [], []

#    ginis = np.array(ginis, dtype=object)
#    gini_best = max(ginis)
#    threshold_best = thresholds[ginis.argmax()]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None,
                 min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical",
                           feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click/current_count
                sorted_categories = list(map(lambda x: x[0],
                                             sorted(ratio.items(),
                                                    key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories,
                                          list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(feature_vector) == 0:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, np.array(sub_y))
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold,
                                                     categories_map.items())))
                else:
                    raise ValueError

        if gini_best == -1:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        return

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
