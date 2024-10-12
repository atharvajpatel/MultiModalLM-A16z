'''
MAML training explanation (20% on just multimodal images)
    Use a learning rate scheduler suitable for meta learning → exponential decay scheduler
        Hyperparameters 
            Inner and Outer loop lr
            Batch size
            Gradient clipping
    Step 1 = Inner loop
        Used the train weights and essentially finetune them for each given category (Personal well being, social and ….)
    Step 2 = Outer loop
        “Meta update” the main model’s parameters based on the final fine tuned alterations found in the inner loop
        Should result in altered model weights that will perform well in the “personal well being, social …:” classes
    Not really “different datasets” but am learning 3 different MAML categories from one distribution (it's the only multimodal option so I broke it into these categories so it finetunes from different angles
        Personal Well-being and Emotions
        Social Interactions and Relationships
        Activities, Entertainment, and Experiences
    Implement model checkpointing and early stopping
        Will model checkpoint at every 2 epochs (planning for 10) so 5 different checkpoints
    Early stopping should occur at 95% accuracy

'''