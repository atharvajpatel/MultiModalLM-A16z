'''
Vanilla sentiment analysis
Step 1 = Tokenize the data
Step 2 = Set pretrained BERT to learn the sentiment by fine tuning it on reddit/twitter and sentiment social media posts dataset.
No need for train test dev as its pretrained BERT and is less data BERT could use. Already have supervised examples so rather pass more for more accurate fine-tune
'''

#import statements
#load in the other posts_final and reddit+twitter_final.csv (define both paths)
#combine them both
#Load in distilbert that can be finetuned (Not frozen version)
#finetune this bert model on all examples
#save the model weights as a .pth file
#test on just text of multimodal_final.csv (only load text and sentiment and calculate accuracy)

