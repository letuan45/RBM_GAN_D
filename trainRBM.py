import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from rbm import RBM
import torchvision.transforms as T
from CustomDataset import IotDataset
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

########## Train RBM ##############
###################################

########## CONFIGURATION ##########
BATCH_SIZE = 64
#VISIBLE_UNITS = 784 # 28 x 28 images
VISIBLE_UNITS = 4096 # 64 x 64 images
HIDDEN_UNITS = 256#128
CD_K = 2 #2
EPOCHS = 20

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0

img_height = 64
img_width = 64


if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

########## LOADING DATASET ##########
print('Loading dataset...')

#Đọc data set ảnh có kích thước 28x28, ảnh xám (1 kênh màu)
full_dataset = IotDataset(csv_file='full_dataset.csv', root_dir='DatasetImages')

#Chia tỉ lệ dataset 75 train - 25 test
train_size = int(0.75 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

#Khởi tạo dataloader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)


######### TRAINING RBM ##########
print('Training RBM...')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)

for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)#flatten input data

        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch)

        epoch_error += batch_error

    print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

########### PHÂN TÁCH ĐẶC TRƯNG ##########
print('Extracting features...')

train_features = np.zeros((len(train_dataset), HIDDEN_UNITS))
train_labels = np.zeros(len(train_dataset))
test_features = np.zeros((len(test_dataset), HIDDEN_UNITS))
test_labels = np.zeros(len(test_dataset))

for i, (batch, labels) in enumerate(train_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    train_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    train_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()

for i, (batch, labels) in enumerate(test_loader):
    batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data

    if CUDA:
        batch = batch.cuda()

    test_features[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = rbm.sample_hidden(batch).cpu().numpy()
    test_labels[i*BATCH_SIZE:i*BATCH_SIZE+len(batch)] = labels.numpy()


########## CLASSIFICATION ##########
print('Classifying...')

model = LogisticRegression(max_iter=3000)
model.fit(train_features, train_labels) #fit với đặc trưng
predictions = model.predict(test_features)

print('SAVING MODEL...')
# save the model to disk
filename = 'rbm_model.sav'
pickle.dump(model, open(filename, 'wb'))

print('Result: %d/%d' % (sum(predictions == test_labels), test_labels.shape[0]))
#print("Model accuracy: {:.2f}".format(model.score(test_features, test_labels)))

matrix_confusion = confusion_matrix(test_labels, predictions)

plt.figure(figsize=(10,5))
plt.title('Ma Trận nhầm lẫn model RBM')
sns.heatmap(matrix_confusion, annot=True, fmt="d", linewidths=.5)
plt.show()

#Feature impotance
# giá trị dương đặc trưng cho malware (nhãn 1)
# giá trị âm đặc trưng cho benign (nhãn 0)
importance = model.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
 print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

#Lấy các TP TN FP FN từ ma trận nhầm lẫn
TP, FP, FN, TN = matrix_confusion[0][0], matrix_confusion[0][1], matrix_confusion[1][0], matrix_confusion[1][1]

print("Accuracy RBM: ", accuracy_score(test_labels, predictions))

#Sensitivity, Recall, Hit Rate, Or True Positive Rate (TPR)
TPR = TP / (TP + FN)
print("TPR: ", TPR)

#Positive Predictive Value (PPV)
PPV = TP / (TP + FP)
print("PPV: ", PPV)

#NegativePredictive Value  (NPV)
NPV = TN / (TN + FN)
print("NPV: ", NPV)

#False Negative Rate (FNR)
FNR = FN / (TP + FN)
print("FNR: ", FNR)

#False Positive Rate (FPR)
FPR = FP / (TN + FP)
print("FPR: ", FPR)

#False Discovery Rate(FDR)
FDR = FP / (FP + TP)
print("FDR: ", FDR)

#False Omission Rate (FOR)
FOR = FN / (FN + TN)
print("FOR: ", FOR)

#F1 score 
F1 = (2 * PPV * TPR) / (PPV + TPR)
print("F1: ", F1)