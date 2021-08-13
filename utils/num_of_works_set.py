import time
import torch.utils.data as d
import torchvision
import torchvision.transforms as transforms

if __name__ == '__main__':
    BATCH_SIZE = 100
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.MNIST('\mnist', download=True, train=True, transform=transform)

    # data loaders
    train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    for num_workers in range(20):
        train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
        # training ...
        start = time.time()
        for epoch in range(1):
            for step, (batch_x, batch_y) in enumerate(train_loader):
                pass
        end = time.time()
        print('num_workers is {} and it took {} seconds'.format(num_workers, end - start))