from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import io, random, base64
import numpy, matplotlib, pandas
import sklearn.ensemble

def random_forest(request):
    return render(request, 'random_forest/random_forest.html')

def random_forest_play(request):
    # Importing the dataset
    file = staticfiles_storage.path('xlsx/Social_Network_Ads.csv')
    dataset = pandas.read_csv(file)
    # dataset = pandas.read_csv('./static/xlsx/Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    # Setting split random state
    if request.POST:
        try:
            curr_rdm = int(request.POST['curr_rdm'])
        except:
            pass
    else:
        curr_rdm = random.randint(0, 9999)
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.25, random_state = curr_rdm)
    # Feature Scaling
    sc = sklearn.preprocessing.StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Fitting classifier to the Training set
    classifier = sklearn.ensemble.RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Making the Confusion Matrix
    # cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    # Assigning the output variables
    user_values = []
    if request.POST:
        try:
            user_age = int(request.POST['user_age'])
            user_income = int(request.POST['user_income'].replace(',', ''))
            user_pred = classifier.predict(sc.transform([[user_age, user_income]]))
            if user_pred == 0:
                user_pred = 'No'
            else:
                user_pred = 'Yes'
            user_values = (user_age, user_income, user_pred)
        except:
            pass
    all_values = []
    for i in range(X.__len__()):
        all_values += [[X[i, 0], X[i, 1], 'No' if y[i] == 0 else 'Yes']]
    original_X_train = sc.inverse_transform(X_train)
    train_values = []
    for i in range(X_train.__len__()):
        train_values += [[int(original_X_train[i, 0]), int(original_X_train[i, 1]), 'No' if y_train[i] == 0 else 'Yes']]
    original_X_test = sc.inverse_transform(X_test)
    test_values = []
    for i in range(X_test.__len__()):
        test_values += [[int(original_X_test[i, 0]), int(original_X_test[i, 1]), 'No' if y_test[i] == 0 else 'Yes', 'No' if y_pred[i] == 0 else 'Yes']]
    # Visualising All Data
    matplotlib.pyplot.clf()
    X_set, y_set = X, y
    for i, j in enumerate(numpy.unique(y_set)):
        matplotlib.pyplot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                                  c = matplotlib.colors.ListedColormap(('red', 'green'))(i), label = 'Yes' if j == 1 else 'No')
    matplotlib.pyplot.xlabel('Age')
    matplotlib.pyplot.ylabel('Income')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.tight_layout()
    buf = io.BytesIO()
    matplotlib.pyplot.savefig(buf, format='png')
    buf.seek(0)
    b64_all = base64.b64encode(buf.read()).decode()
    buf.close()
    # Visualising the Training set results
    matplotlib.pyplot.clf()
    X_set, y_set = X_train, y_train
    X1, X2 = numpy.meshgrid(numpy.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                            numpy.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    matplotlib.pyplot.contourf(X1, X2, classifier.predict(numpy.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                               alpha = 0.4, cmap = matplotlib.colors.ListedColormap(('red', 'green')))
    matplotlib.pyplot.xlim(X1.min(), X1.max())
    matplotlib.pyplot.ylim(X2.min(), X2.max())
    for i, j in enumerate(numpy.unique(y_set)):
        matplotlib.pyplot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                                  c = matplotlib.colors.ListedColormap(('red', 'green'))(i), label = 'Yes' if j == 1 else 'No')
    matplotlib.pyplot.xlabel('Age (scaled)')
    matplotlib.pyplot.ylabel('Income (scaled)')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.tight_layout()
    buf = io.BytesIO()
    matplotlib.pyplot.savefig(buf, format='png')
    buf.seek(0)
    b64_train = base64.b64encode(buf.read()).decode()
    buf.close()
    # Visualising the Test set results
    matplotlib.pyplot.clf()
    X_set, y_set = X_test, y_test
    X1, X2 = numpy.meshgrid(numpy.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                            numpy.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    matplotlib.pyplot.contourf(X1, X2, classifier.predict(numpy.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                               alpha = 0.4, cmap = matplotlib.colors.ListedColormap(('red', 'green')))
    matplotlib.pyplot.xlim(X1.min(), X1.max())
    matplotlib.pyplot.ylim(X2.min(), X2.max())
    for i, j in enumerate(numpy.unique(y_set)):
        matplotlib.pyplot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                                  c = matplotlib.colors.ListedColormap(('red', 'green'))(i), label = 'Yes' if j == 1 else 'No')
    if user_values:
        try:
            matplotlib.pyplot.scatter(sc.transform([[user_age, user_income]])[0][0], sc.transform([[user_age, user_income]])[0][1], color = 'yellow', label = 'User')
        except:
            pass
    matplotlib.pyplot.xlabel('Age (scaled)')
    matplotlib.pyplot.ylabel('Income (scaled)')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.tight_layout()
    buf = io.BytesIO()
    matplotlib.pyplot.savefig(buf, format='png')
    buf.seek(0)
    b64_test = base64.b64encode(buf.read()).decode()
    buf.close()
    context = {
        'all_values': all_values,
        'train_values': train_values,
        'test_values': test_values,
        'user_values': user_values,
        'curr_rdm': curr_rdm,
        'b64_train': b64_train,
        'b64_test': b64_test,
        'b64_all': b64_all
    }
    return render(request, 'random_forest/random_forest_play.html', context)