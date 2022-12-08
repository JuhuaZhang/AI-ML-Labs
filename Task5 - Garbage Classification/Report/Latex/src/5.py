transform = transforms.Compose([
    #size transformation
    transforms.Resize((128,128)),
    #rotation       
    transforms.RandomRotation((30,30)), 
    #vertical flip
    transforms.RandomVerticalFlip(0.1),
    #Grayscale 
    transforms.RandomGrayscale(0.1),
    #transform tensor
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))))