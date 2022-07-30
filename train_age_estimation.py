from torch.utils.data import DataLoader
from tqdm import trange
# from ordinal_regression.network import AgeNet
from Age_Estimation.model import MultipleOutputCNN
from ordinal_regression.network import AgeNet
from Age_Estimation.utils import *
from dataset.AFAD_dataset import AFAD

'''
Hyper-para
'''
save_path = ''
epoch = 200
learning_rate = 1e-2
batch_size = 1024
split = np.array([7, 2, 1])
shuffle_dataset = True
random_seed = 42
np.random.seed(random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
l2_loss_fn = torch.nn.MSELoss()
'''
Split the dataset
'''
full_dataset = AFAD()
train_dataset, test_dataset, val_dataset = split_dataset(full_dataset=full_dataset, split=split, is_val=True)
train_dataset.dataset.is_train = True
train_dataset.dataset.Image_Transform()
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
)
if val_dataset:
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )


def val_loop(model, loader, device):
    total = len(loader.dataset)
    mae = 0
    for step, batch in enumerate(loader):
        x, label, age = batch
        x, label = x.double(), label.double()
        x = x.to(device)
        label = label.to(device)
        age = age.to(device)
        predict = model(x)
        mae += MAE(predict, age) * len(age)
    mae = mae / total
    print('validate|| MAE:{:.5f}'.format(mae))
    return mae


def train_loop(model, loader, optimizer, loss_func, device, importance):
    total = len(loader.dataset)
    importance = importance.to(device)
    for step, batch in enumerate(loader):
        x, label, age = batch
        x, label = x.double(), label.double()
        x = x.to(device)
        label = label.to(device)
        age = age.to(device)
        predict = model(x)
        # loss = loss_func(predict, label, importance).to(device)
        loss = l2_loss_fn(predict, label).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mae = MAE(predict, age)
        print('training || loss:{:.7f} MAE:{:.5f} [{}/{}]'.format(loss.item(), mae, len(x) * (step + 1), total))
        # if step % 5 == 0:
        #     print('training || loss:{:.7f} MAE:{:.5f} [{}/{}]'.format(loss.item(), mae, len(x) * (step + 1), total))
    pass


def main():
    # model = MultipleOutputCNN().double()
    model = AgeNet().double()
    model.to(device)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5, last_epoch=-1, verbose=False)
    best_MAE = 72. - 15.
    is_best = 0
    importance = make_task_importance()
    for i in trange(epoch):
        print('-----------------------epoch {}-----------------------'.format(i + 1))
        print('-----------current learning rate: {:.6f}-----------'.format(
            optimizer.state_dict()['param_groups'][0]['lr']))
        model.train()
        train_loop(model, train_dataloader, optimizer, importance_cross_entropy, device, importance)
        with torch.no_grad():
            model.eval()
            mae = val_loop(model, val_dataloader, device)
        if mae < best_MAE:
            best_MAE = mae
            is_best = 1
        save_model(model, save_path, 'epoch_{}.pth'.format(i + 1), is_best)
        scheduler.step()
        is_best = 0
    pass


if __name__ == '__main__':
    main()
