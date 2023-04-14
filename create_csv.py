import csv
import os

def read_data(path,label):
    images = []
    labels = []
    dirs = os.listdir( path )
    for i, files in enumerate(dirs):
        file_name = os.path.basename(files)
        images.append(file_name)
        labels.append(label)
    return images, labels

image_mal, lable_mal = read_data('out/malware', 1)
image_ben, lable_ben = read_data('out/benign', 0)

images = image_mal + image_ben
lables = lable_mal + lable_ben

# csv data
data_rows = []
for i in range(len(images)):
    data_rows.append({'filename': images[i], 'lable': lables[i]})
    
# csv header
fieldnames = ['filename', 'lable']

with open('full_dataset.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data_rows)