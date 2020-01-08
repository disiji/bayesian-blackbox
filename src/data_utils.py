datafile_dict = {
    'cifar100': '../data/cifar100/cifar100_predictions_dropout.txt',
    'imagenet': '../data/imagenet/resnet152_imagenet_outputs.txt',
    'imagenet2_topimages': '../data/imagenet/resnet152_imagenetv2_topimages_outputs.txt',
    '20newsgroup': '../data/20newsgroup/bert_20_newsgroups_outputs.txt',
    'svhn': '../data/svhn/svhn_predictions.txt',
    'dbpedia': '../data/dbpedia/bert_dbpedia_outputs.txt',
}

datasize_dict = {
    'cifar100': 10000,
    'imagenet': 50000,
    'imagenet2_topimages': 10000,
    '20newsgroup': 5607,
    'svhn': 26032,
    'dbpedia': 70000,
}

num_classes_dict = {
    'cifar100': 100,
    'imagenet': 1000,
    'imagenet2_topimages': 1000,
    '20newsgroup': 20,
    'svhn': 10,
    'dbpedia': 14,
}

output_str_dict = {
    'weighted_pool_bayesian_estimation_error': 'weighted_pool_error_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'weighted_pool_frequentist_estimation_error': 'weighted_pool_error_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'weighted_online_bayesian_estimation_error': 'weighted_online_error_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'weighted_online_frequentist_estimation_error': 'weighted_online_error_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'unweighted_bayesian_estimation_error': 'unweighted_error_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'unweighted_frequentist_estimation_error': 'unweighted_error_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'pool_bayesian_ece': 'pool_ece_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'pool_frequentist_ece': 'pool_ece_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'online_bayesian_ece': 'online_ece_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'online_frequentist_ece': 'online_ece_%s_PseudoCount%.1f_runs%d_frequentist.csv',
    'bayesian_mce': 'mce_%s_PseudoCount%.1f_runs%d_bayesian.csv',
    'frequentist_mce': 'mce_%s_PseudoCount%.1f_runs%d_frequentist.csv'
}

DATASET_LIST = ['imagenet', 'dbpedia', 'cifar100', '20newsgroup', 'svhn', 'imagenet2_topimages']
