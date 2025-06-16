# # # import numpy as np
# # # import pickle
# # # from sklearn.preprocessing import LabelEncoder
# # #
# # # # Define all sign language letters (adjust as needed)
# # # labels = ["A", "B", "C", "D", "E"]  # Add all gesture labels from your dataset
# # #
# # # # Initialize Label Encoder and fit labels
# # # label_encoder = LabelEncoder()
# # # label_encoder.fit(labels)
# # #
# # # # Save the label encoder
# # # with open("models/label_encoder.npy", "wb") as f:
# # #     pickle.dump(label_encoder, f)
# # #
# # # print("New label encoder saved successfully!")
# #
# #
# # import numpy as np
# # import pickle
# # from sklearn.preprocessing import LabelEncoder
# #
# # # Define all gesture labels (Update based on your dataset)
# # labels = ["A", "B", "C", "D", "E"]  # Add all sign language letters in your dataset
# #
# # # Initialize and fit LabelEncoder
# # label_encoder = LabelEncoder()
# # label_encoder.fit(labels)
# #
# # # Save the new label encoder
# # with open("models/label_encoder.npy", "wb") as f:
# #     pickle.dump(label_encoder, f)
# #
# # print("✅ New label encoder saved successfully!")
#
#
# import numpy as np
# import pickle
#
# # Replace with the full list of labels used during training
# labels = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
#
# # Save the new label encoder
# with open("models/label_encoder.npy", "wb") as f:
#     pickle.dump(labels, f)
#
# print("✅ New label encoder saved successfully!")



import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Define all labels used in training
labels = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

# Create and fit LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Save the LabelEncoder object instead of just the NumPy array
with open("models/label_encoder.npy", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ New label encoder saved successfully!")
