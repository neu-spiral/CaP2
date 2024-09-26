import os
import  torch
from    torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import pickle 
def get_dataset_from_code(code, batch_size):
    """ interface to get function object
    Args:
        code(str): specific data type
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    dataset_root = "./assets/data"
    if code == 'mnist':
        train_loader, test_loader = get_mnist_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'mnist-data'))
    elif code == 'cifar10':
        train_loader, test_loader = get_cifar10_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'cifar10-data'))
    elif code == 'fmnist':
        train_loader, test_loader = get_fasionmnist_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'fasionmnist-data'))
    elif code == 'cifar100':
        train_loader, test_loader = get_cifar100_data(batch_size=batch_size,
            data_folder_path=os.path.join(dataset_root, 'cifar100-data'))
    elif code == 'flash':
        train_loader, test_loader = get_flash_data(batch_size=batch_size,
            data_folder_path='/raid/yarkin/cap/data/FLASH_Dataset_3_Processed')
    elif code == 'esc':
        train_loader, test_loader = get_esc_data(batch_size=batch_size,
            data_folder_path='/raid/yarkin/cap/data/ImageDataset')
    else:
        # raise ValueError("Unknown data type : [{}] Impulse Exists".format(data_name))
        raise ValueError("Unknown data type : [{}] Impulse Exists".format(code))

    return train_loader, test_loader


def get_fasionmnist_data(data_folder_path, batch_size=64):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                   #transforms.Normalize((0.2860,), (0.3530,)),
                                 ])
    # Download and load the training data
    trainset = datasets.FashionMNIST(data_folder_path, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.FashionMNIST(data_folder_path, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

def get_mnist_data(data_folder_path, batch_size=64):
    """ mnist data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    train_data = datasets.MNIST(data_folder_path, train=True,  download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    test_data  = datasets.MNIST(data_folder_path, train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_cifar10_data(data_folder_path, batch_size=64):
    """ cifar10 data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(data_folder_path, train=True, 
        download=True, transform=transform_train)
    test_data  = datasets.CIFAR10(data_folder_path, train=False, 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_cifar100_data(data_folder_path, batch_size=64):
    """ cifar100 data
    Args:
        train_batch_size(int): training batch size 
        test_batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    """
    transform_train = transforms.Compose([

        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),

    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_data = datasets.CIFAR100(data_folder_path, train=True, 
        download=True, transform=transform_train)
    test_data  = datasets.CIFAR100(data_folder_path, train=False, 
        download=True, transform=transform_test) 

    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_data, 
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader

def get_esc_data(data_folder_path, batch_size=64):
    ''' ESC data
    Args:
        data_folder_path: data path
        batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    '''

    file_names = ['ESC1', 'ESC2', 'ESC3', 'ESC4', 'ESC5']
    selections = ['Radar', 'NoRadar']
    frame_count = 500

    # with open (data_folder_path + 'esc.pkl','rb') as handle:
    #     data = pickle.load(handle)
    # data_train, data_test = data['train'], data['test']
    
    xtrain,xtest = (),()
    ytrain,ytest = (),()
    # ytrain,ytest = data_train[-1],data_test[-1]

    for file_name in file_names:
        train_input, test_input = [], []
        train_label, test_label = np.array([]), np.array([])
        for selection in selections:
            print('file_name, selection:', file_name, selection)
            if selection == 'Radar':
                frames_total = np.random.permutation(np.arange(1, int(frame_count/2) + 1))
            else:
                frames_total = np.random.permutation(np.arange(int(frame_count/2) + 1 ,frame_count + 1))
            frames_train = frames_total[:int(frames_total.shape[0]*0.8)]
            frames_test = frames_total[int(frames_total.shape[0]*0.8):]
            # name the paths in format of Signals_ESC4_frame_498_NoRadar.jpg
            paths_train = [data_folder_path + '/Signals_' + file_name + '_frame_' + str(i) + '_' + selection + '.jpg' for i in frames_train]
            paths_test = [data_folder_path + '/Signals_' + file_name + '_frame_' + str(i) + '_' + selection + '.jpg' for i in frames_test]
            print(len(paths_train))
            print(len(paths_test))

            train_input = train_input + paths_train
            test_input = test_input + paths_test

            if selection == 'Radar':
                train_label = np.concatenate((train_label, np.ones(len(paths_train), dtype=int)))
                test_label = np.concatenate((test_label, np.ones(len(paths_test), dtype=int)))
            else:
                train_label = np.concatenate((train_label, np.zeros(len(paths_train), dtype=int)))
                test_label = np.concatenate((test_label, np.zeros(len(paths_test), dtype=int)))

        xtrain += (fetch_esc_data(train_input),)
        xtest += (fetch_esc_data(test_input),)

        # xtrain = np.concatenate((xtrain, fetch_esc_data(train_input)))
        # xtest = np.concatenate((xtest, fetch_esc_data(test_input)))

        # print('xtrain shape:', len(xtrain[0]))
        # print(xtrain)

        ytrain = np.concatenate((ytrain, train_label))
        ytest = np.concatenate((ytest, test_label))
        # ytrain += (train_label,)
        # ytest += (test_label,)

    xtrain = (np.concatenate(xtrain),)
    xtest = (np.concatenate(xtest),)
    # print('xtrain shape:', xtrain[0].shape)
    # print('xtest shape:', xtest[0].shape)

    # print('ytrain shape:', ytrain.shape)
    # print('ytest shape:', ytest.shape)

    # for i in tqdm(range(len(data_train)-1)):
    #     xtrain += (fetch_esc_data(data_train[i]),)
    #     xtest += (fetch_esc_data(data_test[i]),)

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0,}
    
    training_set = ESCDataLoader(*xtrain, label=ytrain)
    # training_set = ESCDataLoader(xtrain, label=ytrain)
    # print('training_set',training_set)
    # print(training_set.__len__())
    train_loader = torch.utils.data.DataLoader(training_set, **params)
    
    testing_set = ESCDataLoader(*xtest, label=ytest)
    # testing_set = ESCDataLoader(xtest, label=ytest)
    test_loader = torch.utils.data.DataLoader(testing_set, **params)
    return train_loader, test_loader

class ESCDataLoader(object):
    def __init__(self, *ds, label):
        self.ds = ds
        self.label = label

    def __getitem__(self, index):
        x = []
        for i, ds in enumerate(self.ds):
            x.append(torch.from_numpy(self.ds[i][index]))
        # x = torch.from_numpy(self.ds[index])
        label = self.label[index]
        # return x[0],x[1],x[2],x[3],x[4],torch.tensor(label, dtype=torch.int8).type(torch.LongTensor)
        
        return x[0],torch.tensor(label, dtype=torch.int8).type(torch.LongTensor)

    def __len__(self):
        return len(self.ds[0])  # assume both datasets have same length

def fetch_esc_data(data):
    x = []
    for image_path in data:
        image = Image.open(image_path) #(677,532,3)
        # print('image',image.size)
        image = np.asarray(image.resize((320,266)))
        # print('image',image.shape)
        
        # image = image/255.
        # print('image',image.shape)
        image = np.moveaxis(image, -1, 0)
        # print('image',image.shape)
        #dimensions = self.args.input_dims   #(512,512)
        #image = cv2.resize(image, dimensions)
        x.append(image)
    return x
    
def get_flash_data(data_folder_path, batch_size=64):
    ''' FLASH data
    Args:
        data_folder_path: data path
        batch_size(int): test batch size
    Returns:
        (torch.utils.data.DataLoader): train loader 
        (torch.utils.data.DataLoader): test loader
    '''
    experiment_catergories = ['Cat1','Cat2','Cat3','Cat4']
    #experiment_epiosdes = ['0']
    experiment_epiosdes = ['0','1','2','3','4','5','6','7','8','9']
    selected_paths = detecting_related_file_paths(data_folder_path,experiment_catergories,experiment_epiosdes)
    
    # Outputs
    print('******************Getting RF data*************************')
    RF_train, RF_val, RF_test = fetch_flash_data(selected_paths, 'rf', 'rf')
    ytrain, num_classes = custom_label(RF_train, 'one_hot')
    yval, _ = custom_label(RF_val, 'one_hot')
    ytest, _ = custom_label(RF_test, 'one_hot')

    print('RF data shapes on same client', RF_train.shape, RF_val.shape, RF_test.shape)

    # GPS
    print('******************Getting Gps data*************************')
    X_coord_train, X_coord_validation, X_coord_test = fetch_flash_data(selected_paths,'gps','gps')
    X_coord_train, X_coord_test = X_coord_train/9747, X_coord_test/9747
    X_coord_train, X_coord_test = X_coord_train[:,:,None], X_coord_test[:,:,None]
    print('GPS data shapes',X_coord_train.shape, X_coord_test.shape)
    # Image
    print('******************Getting image data*************************')
    X_img_train, X_img_validation, X_img_test = fetch_flash_data(selected_paths,'image','img')
    X_img_train, X_img_test = X_img_train/255, X_img_test/255
    print('image data shapes',X_img_train.shape, X_img_test.shape)
    
    # Lidar
    print('******************Getting lidar data*************************')
    X_lidar_train, X_lidar_validation, X_lidar_test = fetch_flash_data(selected_paths,'lidar','lidar')
    print('lidar data shapes',X_lidar_train.shape, X_lidar_test.shape)
            
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}
    
    training_set = FlashDataLoader(X_coord_train, X_img_train, X_lidar_train, ytrain)
    train_loader = torch.utils.data.DataLoader(training_set, **params)
    
    testing_set = FlashDataLoader(X_coord_test, X_img_test, X_lidar_test, ytest)
    test_loader = torch.utils.data.DataLoader(testing_set, **params)
    return train_loader, test_loader

class FlashDataLoader(object):
    def __init__(self, ds1, ds2, ds3, label):
        self.ds1 = ds1
        self.ds2 = ds2
        self.ds3 = ds3
        self.label = label

    def __getitem__(self, index):
        x1, x2, x3 = self.ds1[index], self.ds2[index],  self.ds3[index]
        label = self.label[index]
        return torch.from_numpy(x1), torch.from_numpy(x2),  torch.from_numpy(x3), torch.from_numpy(label)

    def __len__(self):
        return self.ds1.shape[0]  # assume both datasets have same length
    
def fetch_flash_data(data_paths,modality,key):   # per cat for now, need to add per epside for FL part
    first = True
    for l in tqdm(data_paths):
        # open_file = open_npz(l+'/'+modality+'.npz',key)
        # print('open_file',open_file.shape)
        # print('l',l)
        # print('modality',modality)
        # print(open_file)
        
        # randperm = np.load(l+'/ranperm.npy')
        try:
            open_file = open_npz(l+'/'+modality+'.npz',key)
            randfile = np.arange(open_file.shape[0])
            randperm = np.random.permutation(randfile)
            train_data = np.concatenate((train_data, open_file[randperm[:int(0.8*len(randperm))]]),axis = 0)
            validation_data = np.concatenate((validation_data, open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]),axis = 0)
            test_data = np.concatenate((test_data, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
        except NameError:
            open_file = open_npz(l+'/'+modality+'.npz',key)
            randfile = np.arange(open_file.shape[0])
            randperm = np.random.permutation(randfile)
            train_data = open_file[randperm[:int(0.8*len(randperm))]]
            validation_data = open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]
            test_data = open_file[randperm[int(0.9*len(randperm)):]]
        
    return train_data,validation_data,test_data

def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
                files_list.append(os.path.join(path, file))
    return files_list

def detecting_related_file_paths(path,categories,episodes):
    find_all_paths =['/'.join(a.split('/')[:-1]) for a in show_all_files_in_directory(path,'rf.npz')]     # rf for example
    # print('find_all_paths',find_all_paths)
    selected = []
    for Cat in categories:   # specify categories as input
        for ep in episodes:
            selected = selected + [s for s in find_all_paths if Cat in s.split('/') and 'episode_'+str(ep) in s.split('/')]
    print('Getting {} data out of {}'.format(len(selected),len(find_all_paths)))

    return selected

def open_npz(path,key):
    data = np.load(path)[key]
    return data

def custom_label(y, strategy='one_hot'):
    'This function generates the labels based on input strategies, one hot, reg'
    y_shape = y.shape
    num_classes = y_shape[1]
    if strategy == 'one_hot':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            # logOut = 20*np.log10(thisOutputs)
            max_index = thisOutputs.argsort()[-1:][::-1]  # For one hot encoding we need the best one
            y[i,:] = 0
            y[i,max_index] = 1

    elif strategy == 'reg':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            # logOut = 20*np.log10(thisOutputs)   # old version
            logOut = thisOutputs
            y[i,:] = logOut
    else:
        print('Invalid strategy')
    return y,num_classes