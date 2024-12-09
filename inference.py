'''
Combining the models 
Step 1 = Pass the text and image and encode it (BERT and Image Library)
Pass to the models
Take average of both for now and round up use 0,1,2 (can do more complex relationship after built)
Evaluation metrics
    Implement attribution techniques (saliency maps and integrated gradients) to understand and give more visibility to model decision making
    Try using just image with gibberish text and vice versa
    Create inference pipelines with only 2 of the 3 combined models and compare with all 3 model accuracy as ablation studies
    Compare against baseline (my guess - correct) and check accuracy with SOTA sentiment analysis models from the papers
'''

#Load all the model weights from multimodalModel, MAML, and vanilla sentiment model
#Load all model architectures 
#Ask SOTA to generate sentiment analysis mock set with images (generate stable diffusion images)
#Train/test split it at 70 30 --> pass 30% to other file eval.py
#Run the corresponding models on the input --> should return multimodal: sentiment, MAML: sentiment, vanilla: sentiment (sentiment is 0,1,2)
#Store all values as a dataframe and the actual output as a dataframe
#Create a neural net to learn the best weights for weighted sum average operation:
    #Multimodal, MAML, Vanilla, Actual
    #Create NN for best set of weights to predict the acutal one
#