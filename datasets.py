import json
import pickle
import tarfile
import numpy as np
import os
import tempfile
import urllib.request
import utils
import shutil
import gzip
import subprocess
import csv
import scipy.io as sp

####################################################################################

CHECKPOINT_DIRECTORY = "../../checkpoints"

DATA_MNIST = "data/MNIST/raw"
DATA_DEBD = "data/DEBD"
DATA_FASHION_MNIST = "data/fashion_mnist"
DATA_SVHN = "data/svhn"
DATA_IMDB = "data/imdb"
DATA_WINE = "data/wine"
DATA_CIFAR_10 = "data/cifar_10"
DATA_THEOREM = "data/theorem"
DATA_BINARY_MNIST = "data/binary_mnist"


####################################################################################


def maybe_download(directory, url_base, filename):
	filepath = os.path.join(directory, filename)
	if os.path.isfile(filepath):
		return False

	if not os.path.isdir(directory):
		utils.mkdir_p(directory)

	url = url_base + filename
	_, zipped_filepath = tempfile.mkstemp(suffix='.gz')
	print('Downloading {} to {}'.format(url, zipped_filepath))
	urllib.request.urlretrieve(url, zipped_filepath)
	print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
	print('Move to {}'.format(filepath))
	shutil.move(zipped_filepath, filepath)
	return True


def maybe_download_mnist():
	mnist_files = ['train-images-idx3-ubyte.gz',
				   'train-labels-idx1-ubyte.gz',
				   't10k-images-idx3-ubyte.gz',
				   't10k-labels-idx1-ubyte.gz']

	for file in mnist_files:
		if not maybe_download(DATA_MNIST, 'http://yann.lecun.com/exdb/mnist/', file):
			continue
		print('unzip data/mnist/{}'.format(file))
		filepath = os.path.join(DATA_MNIST, file)
		with gzip.open(filepath, 'rb') as f_in:
			with open(filepath[0:-3], 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)


def load_mnist():
	"""Load MNIST"""

	maybe_download_mnist()

	fd = open(os.path.join(DATA_MNIST, 'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

	fd = open(os.path.join(DATA_MNIST, 'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	train_labels = loaded[8:].reshape((60000)).astype(np.float32)

	fd = open(os.path.join(DATA_MNIST, 't10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

	fd = open(os.path.join(DATA_MNIST, 't10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	test_labels = loaded[8:].reshape((10000)).astype(np.float32)

	train_labels = np.asarray(train_labels)
	test_labels = np.asarray(test_labels)

	return train_x, train_labels, test_x, test_labels


def maybe_download_fashion_mnist():
	mnist_files = ['train-images-idx3-ubyte.gz',
				   'train-labels-idx1-ubyte.gz',
				   't10k-images-idx3-ubyte.gz',
				   't10k-labels-idx1-ubyte.gz']

	for file in mnist_files:
		if not maybe_download(DATA_FASHION_MNIST, 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', file):
			continue
		print('unzip data/fashion_mnist/{}'.format(file))
		filepath = os.path.join(DATA_FASHION_MNIST, file)
		with gzip.open(filepath, 'rb') as f_in:
			with open(filepath[0:-3], 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)


def maybe_download_cifar_10():
	cifar_10_file = 'cifar-10-python.tar.gz'
	if not maybe_download(DATA_CIFAR_10, 'https://www.cs.toronto.edu/~kriz/', cifar_10_file):
		return
	cifar_tar = tarfile.open(os.path.join(DATA_CIFAR_10, cifar_10_file))
	cifar_tar.extractall(DATA_CIFAR_10)
	cifar_tar.close()


def load_cifar_10():
	DATA_CIFAR_FOLDER = os.path.join(DATA_CIFAR_10, 'cifar-10-batches-py')
	train_x = []
	train_labels = []
	test_x = []
	test_labels = []
	cifar_train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
	for file in cifar_train_files:
		filepath = os.path.join(DATA_CIFAR_FOLDER, file)
		with open(filepath, 'rb') as f_in:
			dictionary = pickle.load(f_in, encoding='bytes')
			train_x.append(dictionary[b'data'])
			train_labels.append(dictionary[b'labels'])

	filepath = os.path.join(DATA_CIFAR_FOLDER, 'test_batch')
	with open(filepath, 'rb') as f_in:
		dictionary = pickle.load(f_in, encoding='bytes')
		test_x.append(dictionary[b'data'])
		test_labels.append(dictionary[b'labels'])

	train_x = np.concatenate(train_x).astype(np.float32)
	train_labels = np.concatenate(train_labels).reshape((-1)).astype(np.float32)
	test_x = np.concatenate(test_x).astype(np.float32)
	test_labels = np.concatenate(test_labels).reshape((-1)).astype(np.float32)

	return train_x, train_labels, test_x, test_labels


def load_fashion_mnist():
	"""Load fashion-MNIST"""

	maybe_download_fashion_mnist()

	fd = open(os.path.join(DATA_FASHION_MNIST, 'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

	fd = open(os.path.join(DATA_FASHION_MNIST, 'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	train_labels = loaded[8:].reshape((60000)).astype(np.float32)

	fd = open(os.path.join(DATA_FASHION_MNIST, 't10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

	fd = open(os.path.join(DATA_FASHION_MNIST, 't10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd, dtype=np.uint8)
	test_labels = loaded[8:].reshape((10000)).astype(np.float32)

	train_labels = np.asarray(train_labels)
	test_labels = np.asarray(test_labels)

	return train_x, train_labels, test_x, test_labels


DEBD = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd',
		'kosarek', 'moviereview', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail', 'voting']

DEBD_shapes = {
	'accidents': dict(train=(12758, 111), valid=(2551, 111), test=(1700, 111)),
	'ad': dict(train=(2461, 1556), valid=(491, 1556), test=(327, 1556)),
	'baudio': dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
	'bbc': dict(train=(1670, 1058), valid=(330, 1058), test=(225, 1058)),
	'bnetflix': dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
	'book': dict(train=(8700, 500), valid=(1739, 500), test=(1159, 500)),
	'c20ng': dict(train=(11293, 910), valid=(3764, 910), test=(3764, 910)),
	'cr52': dict(train=(6532, 889), valid=(1540, 889), test=(1028, 889)),
	'cwebkb': dict(train=(2803, 839), valid=(838, 839), test=(558, 839)),
	'dna': dict(train=(1600, 180), valid=(1186, 180), test=(400, 180)),
	'jester': dict(train=(9000, 100), valid=(4116, 100), test=(1000, 100)),
	'kdd': dict(train=(180092, 64), valid=(34955, 64), test=(19907, 64)),
	'kosarek': dict(train=(33375, 190), valid=(6675, 190), test=(4450, 190)),
	'moviereview': dict(train=(1600, 1001), valid=(250, 1001), test=(150, 1001)),
	'msnbc': dict(train=(291326, 17), valid=(58265, 17), test=(38843, 17)),
	'msweb': dict(train=(29441, 294), valid=(5000, 294), test=(3270, 294)),
	'nltcs': dict(train=(16181, 16), valid=(3236, 16), test=(2157, 16)),
	'plants': dict(train=(17412, 69), valid=(3482, 69), test=(2321, 69)),
	'pumsb_star': dict(train=(12262, 163), valid=(2452, 163), test=(1635, 163)),
	'tmovie': dict(train=(4524, 500), valid=(591, 500), test=(1002, 500)),
	'tretail': dict(train=(22041, 135), valid=(4408, 135), test=(2938, 135)),
	'voting': dict(train=(1214, 1359), valid=(350, 1359), test=(200, 1359)),
}

DEBD_display_name = {
	'accidents': 'accidents',
	'ad': 'ad',
	'baudio': 'audio',
	'bbc': 'bbc',
	'bnetflix': 'netflix',
	'book': 'book',
	'c20ng': '20ng',
	'cr52': 'reuters-52',
	'cwebkb': 'web-kb',
	'dna': 'dna',
	'jester': 'jester',
	'kdd': 'kdd-2k',
	'kosarek': 'kosarek',
	'moviereview': 'moviereview',
	'msnbc': 'msnbc',
	'msweb': 'msweb',
	'nltcs': 'nltcs',
	'plants': 'plants',
	'pumsb_star': 'pumsb-star',
	'tmovie': 'each-movie',
	'tretail': 'retail',
	'voting': 'voting'}


def maybe_download_debd():
	if os.path.isdir(DATA_DEBD):
		return
	subprocess.run(['git', 'clone', 'https://github.com/arranger1044/DEBD', DATA_DEBD])
	wd = os.getcwd()
	os.chdir('data/DEBD')
	os.chdir(wd)


def load_debd(name, dtype='int32'):
	"""Load one of the twenty binary density esimtation benchmark datasets."""

	maybe_download_debd()

	train_path = os.path.join(DATA_DEBD, 'datasets', name, name + '.train.data')
	test_path = os.path.join(DATA_DEBD, 'datasets', name, name + '.test.data')
	valid_path = os.path.join(DATA_DEBD, 'datasets', name, name + '.valid.data')

	reader = csv.reader(open(train_path, 'r'), delimiter=',')
	train_x = np.array(list(reader)).astype(dtype)

	reader = csv.reader(open(test_path, 'r'), delimiter=',')
	test_x = np.array(list(reader)).astype(dtype)

	reader = csv.reader(open(valid_path, 'r'), delimiter=',')
	valid_x = np.array(list(reader)).astype(dtype)

	return train_x, test_x, valid_x


def maybe_download_svhn():
	svhn_files = ['train_32x32.mat', 'test_32x32.mat', "extra_32x32.mat"]
	for file in svhn_files:
		maybe_download(DATA_SVHN, 'http://ufldl.stanford.edu/housenumbers/', file)


def load_svhn(dtype=np.uint8):
	"""
    Load the SVHN dataset.
    """

	maybe_download_svhn()

	data_train = sp.loadmat(os.path.join(DATA_SVHN, "train_32x32.mat"))
	data_test = sp.loadmat(os.path.join(DATA_SVHN, "test_32x32.mat"))
	data_extra = sp.loadmat(os.path.join(DATA_SVHN, "extra_32x32.mat"))

	train_x = data_train["X"].astype(dtype).reshape(32 * 32, 3, -1).transpose(2, 0, 1)
	train_labels = data_train["y"].reshape(-1)

	test_x = data_test["X"].astype(dtype).reshape(32 * 32, 3, -1).transpose(2, 0, 1)
	test_labels = data_test["y"].reshape(-1)

	extra_x = data_extra["X"].astype(dtype).reshape(32 * 32, 3, -1).transpose(2, 0, 1)
	extra_labels = data_extra["y"].reshape(-1)

	return train_x, train_labels, test_x, test_labels, extra_x, extra_labels


def maybe_download_binary_mnist():
	maybe_download(DATA_BINARY_MNIST, 'https://github.com/mgermain/MADE/releases/download/ICML2015/',
				   'binarized_mnist.npz')


def load_binarized_mnist_dataset():
	maybe_download_binary_mnist()

	mnist = np.load(os.path.join(DATA_BINARY_MNIST, 'binarized_mnist.npz'))
	train_x = mnist['train_data']
	valid_x = mnist['valid_data']
	test_x = mnist['test_data']

	return train_x, valid_x, test_x


def maybe_download_imdb():
	raw_data_file = os.path.join(DATA_IMDB, 'imdb.npz')
	word_to_index_file = os.path.join(DATA_IMDB, 'imdb_word_index.json')

	if os.path.isfile(raw_data_file) and os.path.isfile(word_to_index_file):
		print('Already exists: {}, {}'.format(raw_data_file, word_to_index_file))
		return

	if not os.path.isdir(DATA_IMDB):
		utils.mkdir_p(DATA_IMDB)

	if not os.path.isfile(word_to_index_file):
		print('Downloading word_to_index file {}.'.format(word_to_index_file))
		urllib.request.urlretrieve(
			'https://s3.amazonaws.com/text-datasets/imdb_word_index.json',
			word_to_index_file)

	if not os.path.isfile(raw_data_file):
		print('Downloading raw data file {}.'.format(raw_data_file))
		urllib.request.urlretrieve('https://s3.amazonaws.com/text-datasets/imdb.npz', raw_data_file)


def maybe_download_higgs():
	if os.path.isfile('data/higgs/HIGGS.csv'):
		print('Already exists: {}'.format('data/higgs/HIGGS.csv'))
		return

	maybe_download('data/higgs', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/', 'HIGGS.csv.gz')

	print('unzip data/higgs/HIGGS.csv.gz')
	with gzip.open('data/higgs/HIGGS.csv.gz', 'rb') as f_in:
		with open('data/higgs/HIGGS.csv', 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)

	os.remove('data/higgs/HIGGS.csv.gz')


def load_imdb():
	data_file = os.path.join(DATA_IMDB, 'imdb-dense-nmf-{}.pkl'.format(200))
	objects = []
	with (open(data_file, "rb")) as openfile:
		while True:
			try:
				objects.append(pickle.load(openfile))
			except EOFError:
				break
	return objects


def maybe_download_all_data():
	print('Downloading dataset -- this might take a while')

	# print()
	# print('MNIST')
	# maybe_download_mnist()
	#
	# print()
	# print('fashion MNIST')
	# maybe_download_fashion_mnist()
	#
	# print()
	# print('20 binary datasets')
	# maybe_download_debd()
	#
	# print()
	# print('SVHN')
	# maybe_download_svhn()
	#
	# print()
	# print('BINARY_MNIST')
	# maybe_download_binary_mnist()
	#
	# print('')
	# print('*** Check for imdb ***')
	# maybe_download_imdb()
	#
	# print('')
	# print('*** Check for theorem ***')
	# maybe_download(DATA_THEOREM, 'https://www.openml.org/data/get_csv/1587932/phpPbCMyg/', 'theorem.csv')
	#
	# print('')
	# print('*** Check for higgs ***')
	# maybe_download_higgs()
	#
	# print('')
	# print('*** Check for wine ***')
	# maybe_download(DATA_WINE, 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/',
	# 			   'winequality-red.csv')
	# maybe_download(DATA_WINE, 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/',
	# 			   'winequality-white.csv')

	print('')
	print("Check for CIFAR-10")
	maybe_download_cifar_10()


############################################################################################################################


def process_imdb(out_path='data/imdb',
				 valid_split=0.1,
				 max_df=0.8,
				 n_words=2000,
				 skip_top=20,
				 max_words=1000,
				 max_topics=200,
				 start_char=1,
				 oov_char=2,
				 rand_gen=1337):
	"""Adopted from keras/datasets/imdb/
    """

	out_file = os.path.join(out_path, 'imdb-dense-nmf-{}.pkl'.format(max_topics))
	if os.path.isfile(out_file):
		print('Already exists: {}'.format(out_file))
		return

	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.decomposition import NMF
	from sklearn.model_selection import train_test_split

	# get word to index dictionary
	word_to_index_file = os.path.join(out_path, 'imdb_word_index.json')
	with open(word_to_index_file) as f:
		word_to_index = json.load(f)

	# get the raw data
	raw_data_file = os.path.join(out_path, 'imdb.npz')
	with np.load(os.path.join(out_path, 'imdb.npz'), allow_pickle=True) as f:
		x_train, labels_train = f['x_train'], f['y_train']
		x_test, labels_test = f['x_test'], f['y_test']

	# pre-processing
	np.random.seed(rand_gen)
	indices = np.arange(len(x_train))
	np.random.shuffle(indices)
	x_train = x_train[indices]
	labels_train = labels_train[indices]

	indices = np.arange(len(x_test))
	np.random.shuffle(indices)
	x_test = x_test[indices]
	labels_test = labels_test[indices]

	xs = np.concatenate([x_train, x_test])
	labels = np.concatenate([labels_train, labels_test])

	if start_char is not None:
		xs = [[start_char] + [w + skip_top for w in x] for x in xs]
	elif skip_top:
		xs = [[w + skip_top for w in x] for x in xs]

	if not n_words:
		n_words = max([max(x) for x in xs])

	if oov_char is not None:
		xs = [[w if (skip_top <= w < n_words) else oov_char for w in x]
			  for x in xs]
	else:
		xs = [[w for w in x if skip_top <= w < n_words]
			  for x in xs]

	idx = len(x_train)
	x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
	x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])
	# end pre-processing

	assert x_train.shape[0] == y_train.shape[0]
	assert x_test.shape[0] == y_test.shape[0]
	print('Loaded dataset imdb with splits:\n\ttrain\t{}\n\ttest\t{}'.format(x_train.shape, x_test.shape))

	x_train, x_valid, y_train, y_valid = train_test_split(
		x_train,
		y_train,
		test_size=valid_split,
		random_state=rand_gen)

	assert x_train.shape[0] == y_train.shape[0]
	assert x_valid.shape[0] == y_valid.shape[0]
	print('Splitted for validation into splits:\n\ttrain\t{}\n\tvalid\t{}\n\ttest\t{}'.format(
		x_train.shape,
		x_valid.shape,
		x_test.shape))

	# translating back to words
	print('Replacing word ids back to tokens {}'.format(x_train[:2]))
	index_to_word = [None] * (max(word_to_index.values()) + 1)
	for w, i in word_to_index.items():
		index_to_word[i] = w

	x_train = [' '.join(index_to_word[i] for i in x_train[i] if i < len(index_to_word)) for i in
			   range(x_train.shape[0])]
	x_valid = [' '.join(index_to_word[i] for i in x_valid[i] if i < len(index_to_word)) for i in
			   range(x_valid.shape[0])]
	x_test = [' '.join(index_to_word[i] for i in x_test[i] if i < len(index_to_word)) for i in range(x_test.shape[0])]

	assert len(x_train) == y_train.shape[0]
	assert len(x_valid) == y_valid.shape[0]
	assert len(x_test) == y_test.shape[0]
	print('Done! {}'.format(x_train[:2]))

	# processing into TF-IDF format
	vectorizer = TfidfVectorizer(lowercase=True,
								 strip_accents='ascii',
								 stop_words='english',
								 max_features=max_words,
								 use_idf=True,
								 max_df=max_df,
								 norm=None)

	vectorizer.fit(x_train)
	x_train = vectorizer.transform(x_train)
	x_valid = vectorizer.transform(x_valid)
	x_test = vectorizer.transform(x_test)

	assert x_train.shape[0] == y_train.shape[0]
	assert x_valid.shape[0] == y_valid.shape[0]
	assert x_test.shape[0] == y_test.shape[0]

	print('TF-IDF shapes:\n\ttrain\t{}\n\tvalid\t{}\n\ttest\t{}'.format(
		x_train.shape,
		x_valid.shape,
		x_test.shape))

	print('After TF-IDF\n {}\n{}\n{}'.format(x_train[:2], x_valid[:2], x_test[:2]))

	data_path = os.path.join(out_path, 'imdb-sparse-tfidf-{}.pklz'.format(x_train.shape[1]))
	with gzip.open(data_path, 'wb') as f:
		pickle.dump((x_train, y_train, x_valid, y_valid, x_test, y_test), f)
	print('Saved to gzipped pickle to {}'.format(data_path))

	nmf = NMF(n_components=max_topics, random_state=rand_gen,
			  beta_loss='kullback-leibler', solver='mu', max_iter=1000,
			  alpha=.1).fit(x_train)

	x_train = nmf.transform(x_train)
	x_valid = nmf.transform(x_valid)
	x_test = nmf.transform(x_test)

	assert x_train.shape[0] == y_train.shape[0]
	assert x_valid.shape[0] == y_valid.shape[0]
	assert x_test.shape[0] == y_test.shape[0]

	print('Final shapes:\n\ttrain\t{}\n\tvalid\t{}\n\ttest\t{}'.format(
		x_train.shape,
		x_valid.shape,
		x_test.shape))

	# saving to pickle
	with open(out_file, 'wb') as f:
		pickle.dump((x_train, y_train, x_valid, y_valid, x_test, y_test), f)
	print('Saved to pickle to {}'.format(out_file))


# return x_train, y_train, x_valid, y_valid, x_test, y_test


def preprocess_wine(path='data/wine', multi_class=False):
	if multi_class:
		out_file = os.path.join(path, 'wine_multiclass.pkl')
	else:
		out_file = os.path.join(path, 'wine.pkl')

	if os.path.isfile(out_file):
		print('Already exists: {}'.format(out_file))
		return

	np.random.seed(1234567890)

	#
	valid_frac = 0.2
	test_frac = 0.2

	wine_red = np.loadtxt(open(path + "/winequality-red.csv", "rb"), delimiter=";", skiprows=1)
	wine_white = np.loadtxt(open(path + "/winequality-white.csv", "rb"), delimiter=";", skiprows=1)

	wine = np.concatenate((wine_red, wine_white))
	wine_x = wine[:, 0:11]
	wine_labels = wine[:, 11]

	for k in range(11):
		print("#{} = {}".format(k, np.sum(wine_labels == k)))

	print("data shape")
	print(wine_x.shape)

	print("first sample")
	print(wine_x[0, :])
	print(wine_labels[0])

	if multi_class:
		wine_labels_ = wine_labels.astype(int)
		wine_labels_[wine_labels <= 4] = 0
		wine_labels_[wine_labels == 5] = 1
		wine_labels_[wine_labels == 6] = 2
		wine_labels_[wine_labels >= 7] = 3
		wine_labels = wine_labels_
	else:
		wine_labels = (wine_labels >= 6).astype(int)

	unique_labels = np.unique(wine_labels)
	print("")
	print("unique labels")
	print(unique_labels)
	for l in unique_labels:
		print("C{}: {}".format(l, 100 * float(np.sum(wine_labels == l)) / float(len(wine_labels))))

	N = wine_x.shape[0]
	valid_N = int(round(N * valid_frac))
	test_N = int(round(N * test_frac))
	train_N = N - (valid_N + test_N)

	rp = np.random.permutation(N)
	train_x = wine_x[rp[0:train_N], :]
	train_labels = wine_labels[rp[0:train_N]]

	valid_x = wine_x[rp[train_N:(train_N + valid_N)], :]
	valid_labels = wine_labels[rp[train_N:(train_N + valid_N)]]

	test_x = wine_x[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]
	test_labels = wine_labels[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]

	print("")
	print("Train shape")
	print(train_x.shape)
	print(train_labels.shape)
	for l in unique_labels:
		print("C{}: {}".format(l, 100 * float(np.sum(train_labels == l)) / float(len(train_labels))))

	print("")
	print("Valid shape")
	print(valid_x.shape)
	print(valid_labels.shape)
	for l in unique_labels:
		print("C{}: {}".format(l, 100 * float(np.sum(valid_labels == l)) / float(len(valid_labels))))

	print("")
	print("Test shape")
	print(test_x.shape)
	print(test_labels.shape)
	for l in unique_labels:
		print("C{}: {}".format(l, 100 * float(np.sum(test_labels == l)) / float(len(test_labels))))

	with open(out_file, 'wb') as f:
		pickle.dump((train_x, train_labels, valid_x, valid_labels, test_x, test_labels), f)


def preprocess_theorem(path='data/theorem'):
	out_file = os.path.join(path, 'theorem.pkl')

	if os.path.isfile(out_file):
		print('Already exists: {}'.format(out_file))
		return

	np.random.seed(101)

	#
	valid_frac = 0.2
	test_frac = 0.2

	theorem = np.loadtxt(open(os.path.join(path, "theorem.csv"), "rb"), delimiter=",", skiprows=1)
	theorem_x = theorem[:, 0:51]
	theorem_labels = theorem[:, 51]

	for k in range(6):
		theorem_labels[theorem_labels == k + 1] = k
	theorem_labels = theorem_labels.astype(int)

	for k in range(10):
		print("#{} = {}".format(k, np.sum(theorem_labels == k)))

	print("data shape")
	print(theorem_x.shape)

	print("first sample")
	print(theorem_x[0, :])
	print(theorem_labels[0])

	unique_labels = np.unique(theorem_labels)
	print("")
	print("unique labels")
	print(unique_labels)
	for l in unique_labels:
		print("C{}: {}".format(l, 100 * float(np.sum(theorem_labels == l)) / float(len(theorem_labels))))

	N = theorem_x.shape[0]
	valid_N = int(round(N * valid_frac))
	test_N = int(round(N * test_frac))
	train_N = N - (valid_N + test_N)

	rp = np.random.permutation(N)
	train_x = theorem_x[rp[0:train_N], :]
	train_labels = theorem_labels[rp[0:train_N]]

	valid_x = theorem_x[rp[train_N:(train_N + valid_N)], :]
	valid_labels = theorem_labels[rp[train_N:(train_N + valid_N)]]

	test_x = theorem_x[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]
	test_labels = theorem_labels[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]

	print("")
	print("Train shape")
	print(train_x.shape)
	print(train_labels.shape)
	for l in unique_labels:
		print("C{}: {}".format(l, 100 * float(np.sum(train_labels == l)) / float(len(train_labels))))

	print("")
	print("Valid shape")
	print(valid_x.shape)
	print(valid_labels.shape)
	for l in unique_labels:
		print("C{}: {}".format(l, 100 * float(np.sum(valid_labels == l)) / float(len(valid_labels))))

	print("")
	print("Test shape")
	print(test_x.shape)
	print(test_labels.shape)
	for l in unique_labels:
		print("C{}: {}".format(l, 100 * float(np.sum(test_labels == l)) / float(len(test_labels))))

	pickle.dump((train_x, train_labels, valid_x, valid_labels, test_x, test_labels), open(out_file, 'wb'))


def preprocess_higgs(path='data/higgs'):
	out_file = os.path.join(path, 'higgs.pkl')
	if os.path.isfile(out_file):
		print('Already exists: {}'.format(out_file))
		return

	valid_N = 1000000
	test_N = 1000000

	higgs = np.loadtxt(open(path + "/HIGGS.csv", "rb"), delimiter=",", skiprows=0)

	higgs_labels = higgs[:, 0].astype(int)
	higgs_x = higgs[:, 1:]

	N = higgs_x.shape[0]
	train_N = N - (valid_N + test_N)

	rp = np.random.permutation(N)
	train_x = higgs_x[rp[0:train_N], :]
	train_labels = higgs_labels[rp[0:train_N]]

	valid_x = higgs_x[rp[train_N:(train_N + valid_N)], :]
	valid_labels = higgs_labels[rp[train_N:(train_N + valid_N)]]

	test_x = higgs_x[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]
	test_labels = higgs_labels[rp[(train_N + valid_N):(train_N + valid_N + test_N)]]

	print(train_x.shape)
	print(train_labels.shape)

	print(valid_x.shape)
	print(valid_labels.shape)

	print(test_x.shape)
	print(test_labels.shape)

	print(np.unique(train_labels))
	print(np.unique(valid_labels))
	print(np.unique(test_labels))

	mu = np.mean(train_x, 0)
	sigma = np.std(train_x, 0)

	train_x = (train_x - mu) / (sigma + 1e-6)
	valid_x = (valid_x - mu) / (sigma + 1e-6)
	test_x = (test_x - mu) / (sigma + 1e-6)

	pickle.dump((train_x, train_labels, valid_x, valid_labels, test_x, test_labels, mu, sigma), open(out_file, 'wb'))


def preprocess_data():
	print('')
	print('*** Preprocess imdb -- this may take some time ***')
	process_imdb()

	print('')
	print('*** Preprocess wine -- this may take some time ***')
	preprocess_wine()

	print('')
	print('*** Preprocess theorem -- this may take some time ***')
	preprocess_theorem()

	print('')
	print('*** Preprocess higgs -- this may take some time ***')
	preprocess_higgs()


if __name__ == '__main__':
	maybe_download_all_data()
	# preprocess_data()
