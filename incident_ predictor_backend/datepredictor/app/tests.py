import os
import pickle
print(pickle.format_version)
import warnings
warnings.filterwarnings("ignore")
# from joblib import dump, load
file1 = os.path.join(r'C:\Users\dilan\OneDrive\Documents\incident_ predictor_backend\datepredictor\app','BestModel.pkl')
with open(file1, 'rb') as pickle_file:
    bestmodel = pickle.load(pickle_file)

# load(file1)