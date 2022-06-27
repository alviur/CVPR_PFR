import torch
from torch import nn
from copy import deepcopy


class LLL_Net(nn.Module):
    """ Basic class for implementing networks """

    def __init__(self, model, remove_existing_head=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        last_layer = getattr(self.model, head_var)

        if remove_existing_head:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                self.model.fc = nn.Sequential()
        else:
            self.out_size = last_layer.out_features

        self.heads = nn.ModuleList()
        self.task_cls = torch.tensor([])
        self.task_offset = []

        self._initialize_weights()

    def add_head(self, num_outputs):
        if len(self.heads):  # we re-compute instead of append in case an approach makes changes to the heads
            self.task_cls = torch.tensor([head.out_features for head in self.heads] + [num_outputs])
        else:
            self.task_cls = torch.tensor(self.task_cls.tolist() + [num_outputs])

        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def remove_all_heads(self):
        self.heads = nn.ModuleList()

    # Simplification to work on multi-head only -- returns all head outputs in a list
    def forward(self, x, return_features=False):
        if len(self.heads) == 0: # only embedding
            return (None, self.embedding(x)) if return_features else self.embedding(x)
        x = self.model(x)
        # assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        for head in self.heads:
            y.append(head(x))
        if return_features:
            return y, x
        else:
            return y

    def embedding(self, x):
        x = self.model(x)
        return x / x.norm(p=2, dim=1, keepdim=True)

    def get_copy(self):
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        # TODO: add the different initializations
        pass
