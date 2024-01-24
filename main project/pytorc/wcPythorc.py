import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



# personal imports
import utils as ut
import sys
import tqdm


# Definisci una semplice rete neurale completamente connessa
class HistogramClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HistogramClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




print("Weather Predictior")


train_paths, test_paths = ut.loadPaths()

index_dataset = 0
train_path, test_path = train_paths[index_dataset], test_paths[index_dataset]

print(f"train_path {train_path}\ntest_paths {test_path}")


""" 
# for command line argouments
# specify if want the path of the dataset
# default path is still ./data
if sys.argv == 2:
    data = ut.loadDataset(sys.argv[1],0)
if sys.argv == 1:
    data = ut.loadDataset()
else:
    print("usage python wc.py path/to/data")
 """

# Carica i dati di addestramento e di test

# list of trasformation of each image
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
    ]
)

print(f"tranform used \n\n {transform}")

# train path needs to be the
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

# compute the histogram for each image


for inputs, labels in train_loader:
    # Calcola l'istogramma per ciascuna immagine nel minibatch
    histograms = torch.tensor([ut.calculate_histogram(np.array(img)) for img in inputs])

    # Appiattisci l'istogramma e convertilo in tensor
    histograms = histograms.view(-1, input_size).float()

    # Azzera i gradienti
    optimizer.zero_grad()

    # Esegui l'inoltro (forward)
    outputs = model(histograms)

    # Calcola la perdita
    loss = criterion(outputs, labels)

    # Esegui la retropropagazione (backward) e l'ottimizzazione
    loss.backward()
    optimizer.step()

exit()
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)


# Imposta parametri della rete e dell'addestramento
input_size = 256 * 256 * 256  # Dimensione dell'istogramma
hidden_size = 128
output_size = 3  # 3 classi di tempo
learning_rate = 0.001

# Inizializza il modello
model = HistogramClassifier(input_size, hidden_size, output_size)
# Definisci la funzione di perdita e l'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Addestra il modello
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Calcola l'istogramma per ciascuna immagine nel minibatch
        histograms = torch.tensor(
            [ut.calculate_histogram(np.array(img)) for img in inputs]
        )

        # Appiattisci l'istogramma e convertilo in tensor
        histograms = histograms.view(-1, input_size).float()

        # Azzera i gradienti
        optimizer.zero_grad()

        # Esegui l'inoltro (forward)
        outputs = model(histograms)

        # Calcola la perdita
        loss = criterion(outputs, labels)

        # Esegui la retropropagazione (backward) e l'ottimizzazione
        loss.backward()
        optimizer.step()


exit()
# Valuta il modello
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # Calcola l'istogramma per ciascuna immagine nel minibatch
        histograms = torch.tensor(
            [calculate_histogram(np.array(img)) for img in inputs]
        )
        histograms = histograms.view(-1, input_size).float()

        # Esegui l'inoltro (forward)
        outputs = model(histograms)

        # Ottieni le previsioni e calcola l'accuratezza
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print("Test Accuracy: {:.2%}".format(accuracy))