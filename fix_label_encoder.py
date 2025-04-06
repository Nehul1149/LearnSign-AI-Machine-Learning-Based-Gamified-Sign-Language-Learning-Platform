# import numpy as np
# from sklearn.preprocessing import LabelEncoder
#
# # Recreate label encoder and fit with labels
# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
#           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
#
# label_encoder = LabelEncoder()
# label_encoder.fit(labels)
#
# # Save correctly
# np.save("models/label_encoder.npy", label_encoder)
# print(" Label encoder saved successfully!")

import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Define the labels for the hand signs
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Initialize LabelEncoder and fit the labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Save the label encoder properly using pickle
with open("models/label_encoder.npy", "wb") as f:
    pickle.dump(label_encoder, f)

print(" Fixed: Label encoder saved successfully!")
