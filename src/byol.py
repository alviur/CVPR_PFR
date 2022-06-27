import numpy as np
import torch
from byol_pytorch import BYOL
from torch import nn
from datasets.data_loader import get_loaders
from networks import resnet32
import utils
from sklearn import preprocessing

device = 'cuda'

utils.seed_everything()
trn_loader, val_loader, tst_loader, taskcla = get_loaders(['cifar100_no_change'], 10, None, 512, 4, False)


train_loader = trn_loader[0]
test_loader = tst_loader[0]

encoder = resnet32(pretrained=True)
encoder.fc = nn.Identity()
encoder.to(device)

# output_feature_dim = encoder.projetion.net[0].in_features
learner = BYOL(
    encoder,
    image_size=32,
    hidden_layer='avgpool',
    use_momentum=False,
    projection_hidden_size=2048
)
learner.to(device)

# opt = torch.optim.Adam(learner.parameters(), lr=0.03)
opt = torch.optim.SGD(learner.parameters(), lr=0.03, weight_decay=0.0005, momentum=0.9)


# set embedding dim
with torch.no_grad():
    output_feature_dim = encoder(trn_loader[0].dataset[0][0].unsqueeze(0).to(device)).shape[-1]
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
for epoch in range(1, 801):
    if epoch in [400, 700]:
        for g in opt.param_groups:
            print(f"Lowering lr: {g['lr']} to {g['lr']/10}")
            g['lr'] = g['lr'] / 10

    run_loss = 0.0
    for images, labels in trn_loader[0]:
        loss = learner(images.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        # learner.update_moving_average()  # update moving average of target encoder
        run_loss += loss.item() / len(labels)

    print(f"Epoch {epoch} - loss: {run_loss}")

    if epoch % 50 == 1:
        encoder.eval()
        x_train, y_train = get_features_from_encoder(encoder, trn_loader[0])
        x_test, y_test = get_features_from_encoder(encoder, tst_loader[0])

        if len(x_train.shape) > 2:
            x_train = torch.mean(x_train, dim=[2, 3])
            x_test = torch.mean(x_test, dim=[2, 3])

        # print("Training data shape:", x_train.shape, y_train.shape)
        # print("Testing data shape:", x_test.shape, y_test.shape)
        scaler = preprocessing.StandardScaler()
        scaler.fit(x_train.cpu().numpy())
        x_train = scaler.transform(x_train.cpu().numpy()).astype(np.float32)
        x_test = scaler.transform(x_test.cpu().numpy()).astype(np.float32)

        train_loader, test_loader = create_data_loaders_from_arrays(
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
            for x, y in train_loader:

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
                for x, y in test_loader:
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
torch.save(encoder.state_dict(), './improved-net.pt')