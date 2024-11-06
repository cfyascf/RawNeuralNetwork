import numpy as np
import pandas as pd

def get_content(file_path):
    data = pd.read_csv(file_path)
    labels = data['Health_Issue']
    features = data.drop('Health_Issue', axis=1)
    
    for column in features.columns:
        if(features[column].dtype == 'object'):
            features[column] = features[column].astype('category').cat.codes
            
    return labels.to_numpy(), features.to_numpy()

# ..DATA..
dataset = 'synthetic_covid_impact_on_work.csv'
labels, columns = get_content(dataset)

# ..CONSTANTS..
epocs = 100000 
# ..each epoc represents the time when all
# the data has been ran throught..

learning_rate = 0.01
# ..keep it low..

patterns = columns.shape[1]
# ..how many features there is 
# to train from..

bias = 1
# ..changes the function's angle..

input_neurons = patterns
hidden_neurons = 128
output_neurons = 1

# ..VARIABLES..
weights1 = np.random.rand(hidden_neurons, input_neurons + 1)
weights2 = np.random.rand(output_neurons, hidden_neurons + 1)

weights1 = weights1 - 0.5
weights2 = weights2 - 0.5
# ..the weights matrix need to have 1 column more
# because the bias is going to be inserted later on..

errors = np.zeros(10000)
errors_mean = np.zeros(epocs)

# ..TRAINING..

# ..for each feature in each epoc..
for i in range(epocs):
    for j in range(10000):
        
        # bias_column = np.full((columns.shape[0], 1), bias)
        # input_biased = np.column_stack((bias_column, columns))
        # input_biased =columns['bias'] = 1
        
        input_biased = np.hstack((bias, columns[j]))
        # ..inserting the bias into the inputs..
        
        output = np.tanh(weights1.dot(input_biased))
        # ..output of first layer X hidden layer..
        
        output_biased = np.insert(output, 0, bias)
        # ..inserting the bias into the output of the first layer..
        
        result = np.tanh(weights2.dot(output_biased))
        # ..output of hidden layer X output layer, the result..
        
        error = labels[j] - result
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
    print(errors_mean[i])
    
def test_model(input):
    input_biased = np.hstack((bias, input))
    output = np.tanh(weights1.dot(input_biased))
    output_biased = np.insert(output, 0, bias)
    result = np.tanh(weights2.dot(output_biased))
    
    return result

test_features = columns[0]
test_label = labels[0]

print(test_model(test_features), test_label)