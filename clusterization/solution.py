import numpy as np

import sklearn
import sklearn.metrics
from sklearn.cluster import KMeans


def silhouette_score(x, labels):
    """
    :param np.ndarray x: Непустой двумерный массив векторов-признаков
    :param np.ndarray labels: Непустой одномерный массив меток объектов
    :return float: Коэффициент силуэта для выборки x с метками labels
    """

    labs, sizes = np.unique(labels, return_counts=True)

    #     print(x)
    #     print(labels)

    if len(labs) == 1:
        return 0

    distances = sklearn.metrics.pairwise_distances(x, force_all_finite=True, n_jobs=-1)

    d = np.zeros((x.shape[0], 1))
    s = np.zeros((x.shape[0], 1))

    for l, size in zip(labs, sizes):
        if size == 1:
            s[labels == l] = 0
            d[labels == l] = 0
            continue

        s[labels == l] = (distances[labels == l][:, labels == l] / (size - 1)).sum(axis=1).reshape((-1, 1))
        matrix = np.hstack(
            [(distances[labels == l][:, labels == c] / sizes[np.where(labs == c)]).sum(axis=1).reshape((-1, 1)) for c in
             labs if c != l])
        d[labels == l] = np.min(matrix, axis=1).reshape((-1, 1))

    s_d_max = np.max(np.hstack([d, s]), axis=1).reshape((-1, 1))
    #     print(s_d_max.shape, d.shape)
    sils = np.where(s_d_max == 0, 0, np.divide(d - s, s_d_max, where=s_d_max != 0))
    #     print(sils)
    sil_score = np.mean(sils)

    return sil_score


def bcubed_score(true_labels, predicted_labels):
    """"
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    """
    #     n = len(true_labels)

    #     L = np.hstack([true_labels.reshape((-1, 1))] * n) == np.vstack([true_labels.reshape((1, -1))] * n)
    #     C = np.hstack([predicted_labels.reshape((-1, 1))] * n) == np.vstack([predicted_labels.reshape((1, -1))] * n)
    L = true_labels[:, np.newaxis] == true_labels[np.newaxis, :]
    C = predicted_labels[:, np.newaxis] == predicted_labels[np.newaxis, :]

    correctness = L * C

    z = correctness.sum(axis=1)

    prec = np.mean(z / C.sum(axis=1))
    recall = np.mean(z / L.sum(axis=1))

    #     prec = np.mean(correctness.sum(axis=1) / C.sum(axis=1))
    #     recall = np.mean(correctness.sum(axis=1) / L.sum(axis=1))

    score = 2 * (prec * recall) / (prec + recall)

    return score


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        """
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.kmeans_ = KMeans(n_clusters=n_clusters, init='k-means++')
        self.mapping_ = None
        self.class_labels_ = None

    def fit(self, data, labels):
        """
            Функция обучает кластеризатор KMeans с заданным числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный объект
        :return KMeansClassifier
        """
        self.kmeans_.fit(data)
        cluster_labels = self.kmeans.labels_

        self.mapping_, self.class_labels_ = self._best_fit_classification(cluster_labels, labels)

        return self

    def predict(self, data):
        """
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        """
        cluster_labels = self.kmeans_.predict(data)

        return self.mapping_[cluster_labels]

    def _best_fit_classification(self, cluster_labels, true_labels):
        """
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом объектов
        внутри этого кластера.
            * Если есть несколько классов с числом объектов, равным максимальному, то выбирается метка с наименьшим номером.
            * Если кластер не содержит размеченных объектов, то выбирается номер класса с максимальным числом элементов в выборке.
            * Если же и таких классов несколько, то также выбирается класс с наименьшим номером
        """
        # подсчет объектов в каждом классе
        uniq_cls_labels, counts = np.unique(true_labels, return_counts=True)

        # убираем неразмеченные объекты
        if uniq_cls_labels[0] == -1:
            uniq_cls_labels = uniq_cls_labels[1:]
            counts = counts[1:]

        # самый большой класс
        biggest_cls_label = uniq_cls_labels[np.argmax(counts)]

        # количетсво классов
        m = len(uniq_cls_labels)

        # матрица формы (n_clusters, m), в клетке [i, j] содержит количество объектов из кластера i и класса uniq_cls_labels[j]
        matrix = np.zeros((self.n_clusters, m))

        #         for i in range(self.n_clusters):
        #             matrix[i] = ((cluster_labels == i)[np.newaxis, :] *
        #                          np.vstack([true_labels == cls for cls in uniq_cls_labels])).sum(axis=1).T
        #         for i in range(self.n_clusters):
        #             matrix[i] = ((cluster_labels == i)[np.newaxis, :] * (true_labels[np.newaxis, :] == uniq_cls_labels[:, np.newaxis])).sum(axis=1).T

        matrix = ((cluster_labels[np.newaxis, :] == np.arange(self.n_clusters)[:, np.newaxis])[:, np.newaxis, :] *
                  (true_labels[np.newaxis, :] == uniq_cls_labels[:, np.newaxis])[np.newaxis, :, :]).sum(axis=-1)

        # вектор соответствий кластеров классам
        mapping = np.zeros((self.n_clusters,))

        mapping = np.where((matrix == 0).all(axis=1), biggest_cls_label, uniq_cls_labels[np.argmax(matrix, axis=1)])

        predicted_labels = mapping[cluster_labels]

        return mapping, predicted_labels
