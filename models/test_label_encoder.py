import pickle

# Load the label encoder
try:
    with open("models/label_encoder.npy", "rb") as f:
        label_encoder = pickle.load(f)

    # Print the classes to check if it's loaded correctly
    print("Label Encoder Classes:", label_encoder.classes_)

except Exception as e:
    print("Error loading label encoder:", str(e))
