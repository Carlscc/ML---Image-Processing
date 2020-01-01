from imageai.Prediction import ImagePrediction
import os
execution_path=os.getcwd() #run from pwd

prediction = ImagePrediction()
prediction.setModelTypeAsSqueezeNet() #set the model type, using SqueezeNet (smaller model size, smaller algorithm - faster prediction and less accuracy)
prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))#set the model path
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "giraffe.jpg"), result_count=5 )#predict the image, with result count 5
for eachPrediction, eachProbability in zip(predictions, probabilities): #zip the predictions and probabilities and print the output
    print(eachPrediction , " : " , eachProbability)