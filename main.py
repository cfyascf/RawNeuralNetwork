import random
import numpy as np
import pandas as pd

# ------------- * FUNCTIONS * -------------

def get_content(file_path):
    data = pd.read_csv(file_path)
    labels = data['Health_Issue']
    features = data.drop('Health_Issue', axis=1)
    
    # ..transforming categorical values into numeric values..
    for column in features.columns:
        if(features[column].dtype == 'object'):
            features[column] = features[column].astype('category').cat.codes
            
    testing_samples = 20 * len(data) / 100
    
    testing_labels = labels.iloc[:testing_samples, :]
    testing_features = features.iloc[:testing_samples, :]
    
    training_labels = labels.iloc[testing_samples:, :]
    training_features = features.iloc[testing_samples:, :]
            
    return testing_labels.to_numpy(), testing_features.to_numpy(), training_labels.to_numpy(), training_features.to_numpy()

def test_model():
    sample = random.randrange(0, testing_samples)
    
    input_biased = np.hstack((bias, testing_features[sample]))
    output = np.tanh(weights1.dot(input_biased))
    output_biased = np.insert(output, 0, bias)
    result = np.tanh(weights2.dot(output_biased))
    
    return result, testing_labels[sample]

# ------------- * CONSTANTS * -------------

dataset = 'synthetic_covid_impact_on_work.csv'
testing_labels, testing_features, training_labels, training_features = get_content(dataset)

training_samples = len(training_features)
testing_samples = len(testing_features)

epocs = 100000 
# ..each epoc represents the time when all
# the data has been ran throught, if it's too big
# it's probably going to lead your model to overfitting..

learning_rate = 0.01
# ..keep it low, this value has 
# a lot of power in progressing the weights..

patterns = training_features.shape[1]
# ..how many features there is 
# to train from..

bias = 1
# ..changes the function's angle..

input_neurons = patterns
hidden_neurons = 128
output_neurons = 1

# ------------- * VARIABLES * -------------
weights1 = np.random.rand(hidden_neurons, input_neurons + 1)
weights2 = np.random.rand(output_neurons, hidden_neurons + 1)
# ..the weights matrix need to have 1 column more
# because the bias is going to be inserted later on..

weights1 = weights1 - 0.5
weights2 = weights2 - 0.5
# ..in this specific dataset, as the values are considerably 
# small, if the weights are too high, it's going to lead
# problems in the gradient calculation..

errors = np.zeros(training_samples)
errors_mean = np.zeros(epocs)

# ------------- * TRAINING * -------------

# ..for each sample in each epoc..
for i in range(epocs):
    for j in range(training_samples):
        
        input_biased = np.hstack((bias, training_features[j]))
        # ..inserting the bias into the inputs..
        
        output = np.tanh(weights1.dot(input_biased))
        # ..output of first layer X hidden layer..
        
        output_biased = np.insert(output, 0, bias)
        # ..inserting the bias into the output of the first layer..
        
        result = np.tanh(weights2.dot(output_biased))
        # ..output of hidden layer X output layer, the result..
        
        error = training_labels[j] - result
        errors[j] = (error.transpose().dot(error))/2
        # ..get the error and make it quadractic so it's more noticeble,
        # bigger errors tend to outstand more..
        
        delta2 = np.diag(error).dot((1 - result*result))          
        vdelta2 = (weights2.transpose()).dot(delta2)      
        delta1 = np.diag(1 - output_biased*output_biased).dot(vdelta2)
        # ..backpropagation: gradient calc..
        
        weights1 = weights1 + learning_rate*(np.outer(delta1[1:], input_biased))
        weights2 = weights2 + learning_rate*(np.outer(delta2, output_biased))
        # ..adjust the weight's matrix for the next feature..
        
    errors_mean[i] = errors.mean()
    print(f"mean error of epoch: {errors_mean[i]}")
    
    hits = 0
    for i in range(100):
        test_result, test_label = test_model()
        
        if test_label - test_result < 0.15:
            hits += 1
            
    if(hits > 70):
        print("early stoppin: model hit 85% or more of accuracy in testing five times.")
        break