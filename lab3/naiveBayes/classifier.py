import os, sys 
import util 

from naiveBayesClassifier import NaiveBayesClassifier
from dataLoader import loadDataset

## Static results for validating classifier output, for samples 20-25 from the contest training and test set
## You can ignore this
STATIC_RESULTS = {
    # dataset
    os.path.join('classifier_data','contest_training.tsv') : {
        # smoothing value
        0 : {
            # log transform
            True :
                [{'West': -9.227803210462996, 'Stop': -9.411029519959127, 'North': -7.915750017879964, 'South': -3.7846703580884906, 'East': -8.558417575623166}, 
                {'West': -5.446010953803324, 'Stop': -10.777600279000344, 'North': -14.613975047788099, 'South': -13.908099207507542, 'East': -6.205623631211526}, 
                {'West': -5.962590405152123, 'Stop': -7.4626872659352, 'North': -8.527562177324732, 'South': -8.122908375965512, 'East': -3.217613093904146}, 
                {'West': -13.878962234614196, 'Stop': -13.991907013378174, 'North': -12.913125065054452, 'South': -12.422082236165704, 'East': -10.80697810272984}, 
                {'West': -6.1951701082620865, 'Stop': -9.627650981053197, 'North': -10.529666884723689, 'South': -9.76250880115709, 'East': -3.807606917198741}],
            False:
              [{'West': 9.826887531876046e-05, 'Stop': 8.181667208040988e-05, 'North': 0.00036495006648903065, 'South': 0.022716349803616382, 'East': 0.00019192275672215022}, 
              {'West': 0.004313477076587676, 'Stop': 2.0861602722669926e-05, 'North': 4.500194432973102e-07, 'South': 9.115683972603913e-07, 'East': 0.0020180498973603666}, 
              {'West': 0.00257323760315477, 'Stop': 0.0005741113069278762, 'North': 0.0001979369173402916, 'South': 0.00029666459236298926, 'East': 0.0400505411417789}, 
              {'West': 9.385194706098227e-07, 'Stop': 8.382855745943637e-07, 'North': 2.465477480787433e-06, 'South': 4.028637871984608e-06, 'East': 2.025764909335204e-05}, 
              {'West': 0.0020392562758453723, 'Stop': 6.588162538516613e-05, 'North': 2.673152747843126e-05, 'South': 5.757000910293849e-05, 'East': 0.022201244853395547}]  
        }, 
        1 : { 
            True : [{'West': -9.224565733640993, 'Stop': -9.5926720730887, 'North': -7.919977495164226, 'South': -3.800993297992063, 'East': -8.558142742015528}, 
            {'West': -5.453350054855271, 'Stop': -10.818397456579895, 'North': -14.508337943967039, 'South': -13.85042132234323, 'East': -6.205388619585945}, 
            {'West': -5.969359838041443, 'Stop': -7.7273550032215805, 'North': -8.503599269400048, 'South': -8.105032447517543, 'East': -3.2286854630405566}, 
            {'West': -13.866173894873832, 'Stop': -13.965764687214174, 'North': -12.876965237672543, 'South': -12.382059022158161, 'East': -10.800597182619805}, 
            {'West': -6.201705564043789, 'Stop': -9.775047846586837, 'North': -10.500720607666135, 'South': -9.740787668269016, 'East': -3.818046931417742}]
        }
    },
    # dataset
    os.path.join('classifier_data','contest_test.tsv') : {
        # smoothing value
        0 : { 
            # log transform 
            True : 
                [{'West': -8.214985753622313, 'Stop': -9.037999687604115, 'North': -6.54643151789153, 'South': -3.833146576695466, 'East': -9.171779349424414}, 
                {'West': -7.151938987181598, 'Stop': -11.58322544298698, 'North': -14.283630570148043, 'South': -14.324592646657184, 'East': -8.542110275881253}, 
                {'West': -6.134670467893881, 'Stop': -8.00501155676056, 'North': -9.825095379731204, 'South': -10.10302528042236, 'East': -3.079603170349666}, 
                {'West': -12.633565074663553, 'Stop': -11.453913465905163, 'North': -9.541701857667059, 'South': -10.830958029581101, 'East': -10.83034605323649}, 
                {'West': -8.236025899851283, 'Stop': -8.933288733515084, 'North': -3.1806208297041123, 'South': -7.998838510793242, 'East': -9.025431749684868}],
            False : 
                [{'West': 0.0002705683656258325, 'Stop': 0.00011880825254229735, 'North': 0.001435228056666375, 'South': 0.02164141201945899, 'East': 0.00010393141480243531}, 
                {'West': 0.0007833437143803091, 'Stop': 9.32114164204401e-06, 'North': 6.261783701143495e-07, 'South': 6.010470335484644e-07, 'East': 0.00019507815674483546}, 
                {'West': 0.0021664390114795673, 'Stop': 0.0003337856435627188, 'North': 5.407733643622679e-05, 'South': 4.0955465848341866e-05, 'East': 0.04597749826395728}, 
                {'West': 3.2607116963005955e-06, 'Stop': 1.0607879691276551e-05, 'North': 7.179455950678126e-05, 'South': 1.9777650318861892e-05, 'East': 1.978975747727844e-05}, 
                {'West': 0.0002649350384771585, 'Stop': 0.000131923449251512, 'North': 0.04155984552372036, 'South': 0.0003358524904908569, 'East': 0.00012031084978068025}]
        },
        # smoothing value
        1: {
            True : [{'West': -8.216596629978746, 'Stop': -9.235752061142371, 'North': -6.552818502787006, 'South': -3.849388383815451, 'East': -9.169071306882143}, 
            {'West': -7.1566260431917526, 'Stop': -11.598556014129471, 'North': -14.178455517024881, 'South': -14.26624921748694, 'East': -8.536699990838388}, 
            {'West': -6.14126748871794, 'Stop': -8.25344809911836, 'North': -9.798807742654704, 'South': -10.079644118285358, 'East': -3.0908156688174357}, 
            {'West': -12.625859065209237, 'Stop': -11.561151831902588, 'North': -9.512684907029238, 'South': -10.794698887230076, 'East': -10.822164279109236}, 
            {'West': -8.237511016732423, 'Stop': -9.138317610756964, 'North': -3.1957966228490453, 'South': -7.994008734879421, 'East': -9.023759052496205}]

        }
    }
}


def staticOutputCheck(dataset, smoothing, logtransform, results):
    if dataset in STATIC_RESULTS: 
        if smoothing in STATIC_RESULTS[dataset]:
            if logtransform in STATIC_RESULTS[dataset][smoothing]:
                # truth
                actual_results = STATIC_RESULTS[dataset][smoothing][logtransform]
                # diff
                total_difference = 0. 

                for groundtruth, prediction in zip(actual_results, results):
                    for key in groundtruth: 
                        total_difference += abs(groundtruth[key] - prediction[key])
                if total_difference < 1e-4:
                    print "The total difference between results of %.5f is inside the allowed range!" % total_difference
                else: 
                    print "The total difference between results of %.5f is too high!" % total_difference

def test():
    """
    Run tests on the implementation of the naive Bayes classifier. 
    The tests are going to be ran on instances 20-25 from both the train and test sets of the contest agent.
    Passing this test is a very good (however not a perfect) indicator that your code is correct.
    """
    train_path = os.path.join('classifier_data', 'contest_training.tsv')
    test_path = os.path.join('classifier_data', 'contest_test.tsv')
    smoothing = [0, 1]
    logtransform = {
        0: [True, False],
        1: [True]
    }
    
    trainData, trainLabels, trainFeatures, = loadDataset(train_path)
    testData, testLabels, testFeatures = loadDataset(test_path)
    
    labels = set(trainLabels) | set(testLabels)
    
    for s in smoothing:
        for lt in logtransform[s]:
            classifierArgs = {'smoothing':s, 'logTransform':lt}
            classifierArgs['legalLabels'] = labels 
            if s:
                featureValues = mergeFeatureValues(trainFeatures, testFeatures) 
                classifierArgs['featureValues'] = featureValues

            # train on train set
            classifier = NaiveBayesClassifier(**classifierArgs)
            classifier.fit(trainData, trainLabels)
            
            # evaluate on train set
            trainPredictions = classifier.predict(trainData)
            evaluateClassifier(trainPredictions, trainLabels, 'train', classifier.k)
            staticOutputCheck(train_path, s, lt, classifier.posteriors[20:25])

            # evaluate on test set
            testPredictions = classifier.predict(testData)
            evaluateClassifier(testPredictions, testLabels, 'test', classifier.k)
            staticOutputCheck(test_path, s, lt, classifier.posteriors[20:25])


def default(str):
    return str + ' [Default: %default]'

def readCommand(argv):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python classifier.py <options>
    EXAMPLES:   >> python classifier.py -t 1 
                    // runs the implementation tests on the 'contest' dataset 
                >> python classifier.py --train contest_training --test contest_test -s 1 -l 1 
                    // run the naive Bayes classifier on the contest dataset with laplace smoothing=1 and log scale transformation
                >> python classifier.py -h 
                    // display the help docstring

    """
    parser = OptionParser(usageStr)

    parser.add_option('--train', dest='train_loc',
                      help=default('the TRAINING DATA for the model'),
                      metavar='TRAIN_DATA', default='stop_training')
    parser.add_option('--test', dest='test_loc',
                      help=default('the TEST DATA for the model'),
                      metavar='TEST_DATA', default='stop_test')
    parser.add_option('-s','--smoothing',dest='smoothing', type='int',
                      help=default('Laplace smoothing'), default=0)
    parser.add_option('-l','--logtransform',dest='logTransform', type='int',
                      help=default('Compute a log transformation of the joint probability'), default=0)
    parser.add_option('-t', '--test_implementation', dest='runtests', type='int',
                      help='Disregard all previous arguments and check if the predicted values match the gold ones', default=0)

    options, otherjunk = parser.parse_args(argv)
    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()

    if options.runtests:
        print ("Running implementation tests:")
        test()
        sys.exit(0)

    # create paths to data files
    data_root = 'classifier_data'
    
    args['train_path'] = os.path.join(data_root, options.train_loc + ".tsv")
    if not os.path.exists(args['train_path']):
        raise Exception("The training data file " + args['train_path'] + " doesn't exist! \n" + 
            "Please only enter the train file name without the extension, and place the file in the" + 
            " 'classifier_data' folder.")

    args['test_path'] = os.path.join(data_root, options.test_loc + ".tsv")
    if not os.path.exists(args['test_path']):
        raise Exception("The test data file " + args['test_path'] + " doesn't exist! \n" + 
            "Please only enter the test file name without the extension, and place the file in the" + 
            " 'classifier_data' folder.")

    args['smoothing'] = options.smoothing 
    args['logtransform'] = options.logTransform != 0 
    return args

def mergeFeatureValues(trainFeatures, testFeatures):
    fullFeatureValues = util.Counter()
    for key in set(trainFeatures.keys() + testFeatures.keys()):
        fullFeatureValues[key] = trainFeatures[key] | testFeatures[key]
    return fullFeatureValues

def evaluateClassifier(predictions, groundtruth, dataset, k):
    accuracy =  [predictions[i] == groundtruth[i] for i in range(len(groundtruth))].count(True)
    print "Performance on %s set for k=%f: (%.1f%%)" % (dataset, k, 100.0*accuracy/len(groundtruth))


def runClassifier(train_path, test_path, smoothing, logtransform):
    classifierArgs = {'smoothing':smoothing, 'logTransform':logtransform}

    trainData, trainLabels, trainFeatures, = loadDataset(train_path)
    testData, testLabels, testFeatures = loadDataset(test_path)
    
    labels = set(trainLabels) | set(testLabels)
    classifierArgs['legalLabels'] = labels 

    if smoothing:
        featureValues = mergeFeatureValues(trainFeatures, testFeatures) 
        classifierArgs['featureValues'] = featureValues

    # train the actual model
    classifier = NaiveBayesClassifier(**classifierArgs)
    classifier.fit(trainData, trainLabels)
    
    trainPredictions = classifier.predict(trainData)
    evaluateClassifier(trainPredictions, trainLabels, 'train', classifier.k)

    testPredictions = classifier.predict(testData)
    evaluateClassifier(testPredictions, testLabels, 'test', classifier.k)


if __name__ == '__main__':
    """
    The main function called when classifeir.py is run
    from the command line:

    > python classifier.py

    See the usage string for more details.

    > python classifier.py --help
    """
    args = readCommand( sys.argv[1:]) # Get game components based on input
    runClassifier(**args)
    pass
