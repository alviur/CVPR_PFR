import numpy as np
import torch
from byol_pytorch import BYOL
from torch import nn
# from datasets.data_loader import get_loaders
from networks import resnet32
import utils
from sklearn import preprocessing
import torchvision
from tqdm import tqdm

device = 'cuda'

BATCH_SIZE = 512
BASE_LR = 0.05

utils.seed_everything()
DATA_DIR = '/data/datasets/cifar10'
train_set = dataset = torchvision.datasets.CIFAR10(
    DATA_DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_set = dataset = torchvision.datasets.CIFAR10(
    DATA_DIR, train=False, download=True, transform=torchvision.transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)

encoder = resnet32()
encoder.fc = nn.Identity()
encoder.to(device)

# output_feature_dim = encoder.projetion.net[0].in_features
learner = BYOL(encoder, image_size=32, hidden_layer='avgpool', use_momentum=False, projection_hidden_size=2048)
learner.to(device)

# multi gpu
# learner = torch.nn.DataParallel(learner)
# if torch.cuda.device_count() > 1:
# learner = torch.nn.SyncBatchNorm.convert_sync_batchnorm(learner)

# opt = torch.optim.Adam(learner.parameters(), lr=0.03)
opt = torch.optim.SGD(learner.parameters(), lr=BASE_LR * BATCH_SIZE / 256, weight_decay=0.0005, momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 801, eta_min=0)

# set embedding dim
with torch.no_grad():
    output_feature_dim = encoder(train_loader.dataset[0][0].unsqueeze(0).to(device)).shape[-1]
    print(f"Embedding dimension: {output_feature_dim}")


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def get_features_from_encoder(encoder, loader):
    x_train = []
    y_train = []
    # get the features from the pre-trained model
    for i, (x, y) in enumerate(loader):
        with torch.no_grad():
            feature_vector = encoder(x.to(device))
            x_train.extend(feature_vector)
            y_train.extend(y.numpy())

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train)
    return x_train, y_train


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):
    train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)
    return train_loader, test_loader


# from task 1
for epoch in tqdm(range(0, 801), desc=f'Training'):
    # if epoch in [400, 700]:
    #     for g in opt.param_groups:
    #         print(f"Lowering lr: {g['lr']} to {g['lr']/10}")
    #         g['lr'] = g['lr'] / 10

    learner.train()
    run_loss = 0.0
    for images, labels in train_loader:
        loss = learner(images.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        # learner.update_moving_average()  # update moving average of target encoder
        run_loss += loss.item() / len(labels)
    lr_scheduler.step()
    print(f"Epoch {epoch} - loss: {run_loss}")

    if epoch % 50 == 1:
        encoder.eval()
        x_train, y_train = get_features_from_encoder(encoder, train_loader)
        x_test, y_test = get_features_from_encoder(encoder, test_loader)

        if len(x_train.shape) > 2:
            x_train = torch.mean(x_train, dim=[2, 3])
            x_test = torch.mean(x_test, dim=[2, 3])

        # print("Training data shape:", x_train.shape, y_train.shape)
        # print("Testing data shape:", x_test.shape, y_test.shape)
        scaler = preprocessing.StandardScaler()
        scaler.fit(x_train.cpu().numpy())
        x_train = scaler.transform(x_train.cpu().numpy()).astype(np.float32)
        x_test = scaler.transform(x_test.cpu().numpy()).astype(np.float32)

        eval_train_loader, eval_test_loader = create_data_loaders_from_arrays(
            torch.from_numpy(x_train), y_train, torch.from_numpy(x_test), y_test
        )

        logreg = LogisticRegression(output_feature_dim, 10)
        logreg = logreg.to(device)
        # optimizer = torch.optim.LBFGS(logreg.parameters(), lr=3e-4)
        optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        eval_every_n_epochs = 10

        for epoch in range(200):
            #     train_acc = []
            for x, y in eval_train_loader:

                x = x.to(device)
                y = y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                logits = logreg(x)
                # predictions = torch.argmax(logits, dim=1)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                # def closure():
                #     # zero the parameter gradients
                #     optimizer.zero_grad()
                #     logits = logreg(x)
                #     # predictions = torch.argmax(logits, dim=1)
                #     loss = criterion(logits, y)
                #     loss.backward()
                #     return loss
                # optimizer.step(closure)

            total = 0
            if epoch % eval_every_n_epochs == 0:
                correct = 0
                for x, y in eval_test_loader:
                    x = x.to(device)
                    y = y.to(device)

                    logits = logreg(x)
                    predictions = torch.argmax(logits, dim=1)

                    total += y.size(0)
                    correct += (predictions == y).sum().item()

                acc = 100 * correct / total
                print(f"Testing accuracy: {np.mean(acc)}")

        encoder.train()
        # END eval

# save your improved network
# torch.save(encoder.state_dict(), './improved-net.pt')