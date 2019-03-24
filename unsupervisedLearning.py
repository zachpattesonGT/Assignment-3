from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import time
import numpy as np
from sklearn import decomposition
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def kMeansClustering(data, mainTitle):
    # This code was originally taken and modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
    start = time.time()
    np.random.seed(5)

    X = data.data
    y = data.target

    fignum = 1
    v_measure_scores = [0, 0, 0, 0, 0, 0, 0]
    ami_scores = [0, 0, 0, 0, 0, 0, 0]
    labels_t = ['2', '3', '4', '5', '6', '7', '8']
    y_pos = np.arange(len(labels_t))
    for x in range(2,9):
        est = KMeans(n_clusters=x)
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2],
                   c=labels.astype(np.float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.dist = 12
        fignum = fignum + 1
        plt.title('K Means with ' + str(x) + ' means: ' + mainTitle)
        fig.savefig('k means ' + str(x) + ' ' + mainTitle + ".png")
        # print("K Means Clustering: " + mainTitle + ' ' + str(x))
        v_measure_scores[x-2] = metrics.v_measure_score(y, est.labels_)
        ami_scores[x-2] = metrics.adjusted_mutual_info_score(y, est.labels_)


    end = time.time()

    plt.clf()
    plt.bar(y_pos, v_measure_scores)
    plt.xlabel('K')
    plt.xticks(y_pos, labels_t)
    plt.title('V Measure Scores: ' + mainTitle)
    plt.savefig('K Means V Measure Scores ' + mainTitle + ".png")

    plt.clf()
    plt.bar(y_pos, ami_scores)
    plt.xlabel('K')
    plt.xticks(y_pos, labels_t)
    plt.title('AMI Scores: ' + mainTitle)
    plt.savefig('K Means AMI Scores ' + mainTitle + ".png")

    print("K Means Clustering: " + mainTitle)
    print("Time: " + str(end-start))
    print('V Measure Score: ' + str(v_measure_scores[2]))
    print('AMI Score: ' + str(ami_scores[2]))


def EM(data, mainTitle):
    # This code was originally taken and modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
    start = time.time()
    np.random.seed(5)

    X = data.data
    y = data.target
    target_names = data.target_names

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    gmm = GaussianMixture(n_components=3)
    gmm.fit(X)
    y_pred = gmm.predict(X)

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,
                   label=target_name)

    ax.legend(loc='best', shadow=False, scatterpoints=1)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.title('EM: ' + mainTitle)
    plt.savefig("EM " + mainTitle + ".png")

    end = time.time()

    print("EM: " + mainTitle)
    print("Time: " + str(end - start))
    print("V Measure Score: " + str(metrics.v_measure_score(y, y_pred)))
    print("AMI Score: " + str(metrics.adjusted_mutual_info_score(y, y_pred)))


def PCA(data, mainTitle):
    # This code was originally taken and modified from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py
    start = time.time()

    np.random.seed(5)

    X = data.data
    y = data.target
    target_names = data.target_names

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components='mle')
    pca.fit(X)
    X = pca.transform(X)

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,
                   label=target_name)

    ax.legend(loc='best', shadow=False, scatterpoints=1)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.title('PCA: ' + mainTitle)
    plt.savefig("PCA " + mainTitle + ".png")

    end = time.time()

    print("PCA: " + mainTitle)
    print("Explained Variance: " + str(pca.explained_variance_ratio_))
    print("Time: " + str(end-start))


def ICA(data, mainTitle):
    # This code was originally taken and modified from https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py
    start = time.time()

    np.random.seed(5)

    X = data.data
    y = data.target
    target_names = data.target_names

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    ica = decomposition.FastICA(n_components=3)
    ica.fit(X)
    X = ica.transform(X)

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,
                   label=target_name)

    ax.legend(loc='best', shadow=False, scatterpoints=1)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.title('ICA: ' + mainTitle)
    plt.savefig("ICA " + mainTitle + ".png")

    end = time.time()

    print("ICA: " + mainTitle)
    print("Time: " + str(end-start))


def RCA(data, mainTitle):
    # This code was originally taken and modified from https://scikit-learn.org/stable/modules/random_projection.html
    start = time.time()

    np.random.seed(5)

    X = data.data
    y = data.target
    target_names = data.target_names

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    rca = random_projection.GaussianRandomProjection(n_components=3)
    rca.fit(X)
    X = rca.transform(X)

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        ax.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,
                   label=target_name)

    ax.legend(loc='best', shadow=False, scatterpoints=1)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.title('RCA: ' + mainTitle)
    plt.savefig("RCA " + mainTitle + ".png")

    end = time.time()

    print("RCA: " + mainTitle)
    print("Time: " + str(end - start))


def LDA(data, mainTitle):
    # This code was originally taken and modified from https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py
    start = time.time()

    np.random.seed(5)

    X = data.data
    y = data.target
    target_names = data.target_names

    pca = decomposition.PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    plt.clf()
    plt.figure()

    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of ' + mainTitle + ' dataset')

    plt.savefig("LDAPCA" + mainTitle + ".png")

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of ' + mainTitle + ' dataset')

    plt.savefig("LDA " + mainTitle + ".png")

    end = time.time()

    print("LDA: " + mainTitle)
    print("Explained Covariance: " + str(lda.explained_variance_ratio_))
    print("Time: " + str(end - start))

def reducedKMeans(reduced_data, data, mainTitle):
    np.random.seed(5)

    y = data.target

    kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
    kmeans.fit(reduced_data)

    fig = plt.figure(0, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    labels = kmeans.labels_

    ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.dist = 12
    plt.title('Reduced K Means: ' + mainTitle)
    fig.savefig('reduced k means ' + mainTitle + ".png")

    v_measure_score = metrics.v_measure_score(y, kmeans.labels_)
    ami_score = metrics.adjusted_mutual_info_score(y, kmeans.labels_)

    print("V Measure Score Reduced " + mainTitle + ": " + str(v_measure_score))
    print("AMI Score Reduced " + mainTitle + ": " + str(ami_score))


def reducedEM(reduced_data, data, mainTitle):
    np.random.seed(5)

    gmm = GaussianMixture(n_components=3)
    gmm.fit(reduced_data)
    y = data.target
    y_pred = gmm.predict(reduced_data)
    target_names = data.target_names

    # fig = plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #
    # colors = ['navy', 'turquoise', 'darkorange']
    # lw = 2
    # for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    #     ax.scatter(reduced_data[:, 0], reduced_data[:, 1], color=color, alpha=.8, lw=lw,
    #                label=target_name)
    #
    # ax.legend(loc='best', shadow=False, scatterpoints=1)
    #
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])

    fig = plt.figure(0, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    ax.scatter(reduced_data[:, 0], reduced_data[:, 1],edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.dist = 12
    plt.title('Reduced K Means: ' + mainTitle)
    fig.savefig('reduced k means ' + mainTitle + ".png")

    v_measure_score = metrics.v_measure_score(y, y_pred)
    ami_score = metrics.adjusted_mutual_info_score(y, y_pred    )

    plt.title('EM: ' + mainTitle)
    plt.savefig("EM " + mainTitle + ".png")

    print("V Measure Score Reduced " + mainTitle + ": " + str(v_measure_score))
    print("AMI Score Reduced " + mainTitle + ": " + str(ami_score))

def mainNeuralNetwork(reduced_data, data, mainTitle):
    # This code was orifinally taken and modified from https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
    start = time.time()
    mainData = data

    X = reduced_data
    y = mainData['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)

    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #mlp = MLPClassifier(hidden_layer_sizes=(X.shape[1], X.shape[1], X.shape[1]))
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        random_state=0)
    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)

    end = time.time()

    print("Neural Network: " + mainTitle)
    print(classification_report(y_test, predictions))
    print("Accuracy: ",
          accuracy_score(y_test, predictions) * 100)
    print("Time: " + str(end - start))

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    title = "Neural Network Learning Curve " + mainTitle
    nnplot = plot_learning_curve(mlp, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    nnplot.savefig("NeuralNetworkLearningCurve" + mainTitle + ".png")

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    # this code was taken from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Calling main function
if __name__ == "__main__":
    dataArray = [load_iris(), load_wine()]
    titleArray = ["Iris", "Wine"]
    for x in range(len(dataArray)):
        kMeansClustering(dataArray[x], titleArray[x])
        EM(dataArray[x], titleArray[x])
        PCA(dataArray[x], titleArray[x])
        ICA(dataArray[x], titleArray[x])
        RCA(dataArray[x], titleArray[x])
        LDA(dataArray[x], titleArray[x])

    for x in range(len(dataArray)):
        pca_reduced_data = decomposition.PCA(n_components=2).fit_transform(dataArray[x].data)
        ica_reduced_data = decomposition.FastICA(n_components=2).fit_transform(dataArray[x].data)
        rca_reduced_data = random_projection.GaussianRandomProjection(n_components=3).fit_transform(dataArray[x].data)
        lda_reduced_data = LinearDiscriminantAnalysis(n_components=2).fit(dataArray[x].data, dataArray[x].target).transform(dataArray[x].data)
        reducedKMeans(pca_reduced_data, dataArray[x], "PCA Reduced " + titleArray[x])
        reducedKMeans(ica_reduced_data, dataArray[x], "ICA Reduced " + titleArray[x])
        reducedKMeans(rca_reduced_data, dataArray[x], "RCA Reduced " + titleArray[x])
        reducedKMeans(lda_reduced_data, dataArray[x], "LDA Reduced " + titleArray[x])
        reducedEM(pca_reduced_data, dataArray[x], "PCA Reduced " + titleArray[x])
        reducedEM(ica_reduced_data, dataArray[x], "ICA Reduced " + titleArray[x])
        reducedEM(rca_reduced_data, dataArray[x], "RCA Reduced " + titleArray[x])
        reducedEM(lda_reduced_data, dataArray[x], "LDA Reduced " + titleArray[x])
        mainNeuralNetwork(pca_reduced_data, dataArray[x], "PCA Reduced " + titleArray[x])
        mainNeuralNetwork(ica_reduced_data, dataArray[x], "ICA Reduced " + titleArray[x])
        mainNeuralNetwork(rca_reduced_data, dataArray[x], "RCA Reduced " + titleArray[x])
        mainNeuralNetwork(lda_reduced_data, dataArray[x], "LDA Reduced " + titleArray[x])