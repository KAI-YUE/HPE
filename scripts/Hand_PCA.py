import os
import pickle
import numpy as np
import random

# sklearn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

src_dir = r"F:\DataSets\SynthHands_Transformed"
X = []

sample_per_folder = 2
total_samples = 4000
already_sampled = 0
for root, dirs, files in os.walk(src_dir):
    if files != []:
        print(root)
        indices = np.arange(len(files))
        random.shuffle(indices)
        for index in indices[:sample_per_folder]:
            with open(os.path.join(root, files[index]), "rb") as fp:
                a_set = pickle.load(fp)            
            norm_3d_pos = a_set["norm_3d_pos"].flatten()[3:]
            X.append(norm_3d_pos.astype("float"))
        already_sampled += sample_per_folder
        if (already_sampled > total_samples):
            break
            

print("Load completed.")            

X = np.asarray(X)
scaler = StandardScaler()
X_ = scaler.fit_transform(X)

pca_model = PCA(n_components=20)
reduced_X = pca_model.fit_transform(X_)
print(pca_model.explained_variance_ratio_)
print(np.sum(pca_model.explained_variance_ratio_))

# weights_arr = pca_model.components_ * scaler.var_
# bias_arr = scaler.mean_

# dat_dict = {"weight":weights_arr.T, "bias":bias_arr}
# with open("D:\\pca(n=20).dat", "wb") as fp:
#     pickle.dump(dat_dict,fp)
