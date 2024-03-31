global_setting = {
    "seed": 3111,
}


model_setting = {
    "embed_size": 300,
    "hidden_size": 256,
    "latent_size": 16,
    "note_size": 88,
    "lstm_layer": 1
}


training_setting = {
    "epochs": 10,
    "batch_size": 32,
    "lr" : 1e-3,
    "clip": 0.25,
    "device": "cpu",
    "train_losses": [],
    "test_losses": []
}