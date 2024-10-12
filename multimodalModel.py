'''
Base Model training explanation (60% train on just multimodal examples)
    Goal —> To be able to compute the sentiment given an image and text 
    Step 1 = Data loading + Train test split
        Use sklearn train test split
        Use torch data loader to standardize text and image
    Step 2 = Model architecture
        Create a MultiModal Class which has both the BERT model and the Vision Transformer as a part of the model 
        Also be able to compute cross attention
    Step 3 = Train
        Use lightning module
            Computations (init).
            Train loop (training_step)
            Validation loop (validation_step)
            Test loop (test_step)
            Optimizers (configure_optimizers)
    Use optimizer and batch/layernorm (most likely need to regularize or do anything fancy to prevent overfitting since very small dataset)
    Use NN.Multihead attention to compute cross attention and then convert to logits and softmax and then calculate loss and go backwards 
    Train options
        1 = Freeze the ViT and BERT weights and only update the NN.Multihead weights which computes cross attention
        2 = Edit the ViT and BERT weights essentially fine tuning
        Will go option 1 since will MAML finetune afterwards anyways 
    Make sure to use gradient accumulation over multiple batches 
    Step 4 = Create inference call for this model (can check on eval/dev set)

'''

#Do all import statements here

#Cuda enable the entire file (should set up with lightning module)