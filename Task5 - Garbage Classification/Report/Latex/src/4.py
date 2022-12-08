# conv1: Conv2d -> BN -> ReLU -> MaxPool
self.conv1 = nn.Sequential(
     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
     nn.BatchNorm2d(16),
     nn.ReLU(), 
     nn.MaxPool2d(kernel_size=2, stride=1),
     )