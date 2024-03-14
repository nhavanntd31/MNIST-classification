from model import Network
from inout import load_mnist, preprocess

if __name__ == '__main__':

    '''
        Hyper parameters
        
            - dataset_name              choose between 'mnist' and 'cifar'
            - num_epochs                number of epochs
            - learning_rate             learning rate
            - validate                  0 -> no validation, 1 -> validation
            - regularization            regularization term (i.e., lambda)
            - verbose                   > 0 --> verbosity
            - plot_weights              > 0 --> plot weights distribution
            - plot_correct              > 0 --> plot correct predicted digits from test set
            - plot_missclassified       > 0 --> plot missclassified digits from test set
            - plot_feature_maps         > 0 --> plot feature maps of predicted digits from test set
    '''

    dataset_name = 'mnist'
    num_epochs = 1
    learning_rate = 0.01
    validate = 1
    regularization = 0
    verbose = 1
    plot_weights = 1
    plot_correct = 0
    plot_missclassified = 0
    plot_feature_maps = 0
    checkpoint_path = "checkpoint"
    print('\n--- Loading ' + dataset_name + ' dataset ---')                 # load dataset
    dataset = load_mnist() 
    print('\n--- Processing the dataset ---')                               # pre process dataset
    dataset = preprocess(dataset)

    print('\n--- Building the model ---')                                   # build model
    model = Network()
    model.build_model(dataset_name)

    model.load_checkpoint("/home/datnt/workspace/CNN-from-Scratch/src (mnist/checkpoint/checkpoint_epoch_1_2024-01-17 22:43:35.250444.npz",dataset_name)
    print('\n--- Training the model ---')                                   # train model
    model.train(
        dataset,
        num_epochs,
        learning_rate,
        validate,
        regularization,
        plot_weights,
        verbose,
        checkpoint_path
    )

    print('\n--- Testing the model ---')                                    # test model
    model.evaluate(
        dataset['test_images'],
        dataset['test_labels'],
        regularization,
        plot_correct,
        plot_missclassified,
        plot_feature_maps,
        verbose
    )
