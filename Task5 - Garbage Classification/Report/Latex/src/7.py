# Train super parameters
config = EasyDict({
    # the dimension of the output layer
    "num_classes": 26, 
    # mean, max, Head part of the pooling method
    "reduction":'mean', 
    "image_height": 224,
    "image_width": 224,
    # In view of the performance of the CPU
    "batch_size": 10, 
    "eval_batch_size": 10,
    "epochs": 50, 
    "lr_max": 0.015, 
    "decay_type":'constant', 
    "momentum": 0.91, 
    "weight_decay": 1e-3, 
    # ... Ingore other parameters
})