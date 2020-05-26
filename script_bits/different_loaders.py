# ---------- Different Loaders and Samplers ----------

# Type 1: ImageFolder with train/valid subfolders, no weights
# (may use shuffle_images() to convert a folder of images into something readable by ImageFolder with train/valid subfolders) 

def build_loader(data_dir,batchsize,train_transforms,valid_transforms):

    train_dir = data_dir+"/train/"
    valid_dir = data_dir+"/valid/"
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batchsize, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = batchsize)
    
    print("Data loaded")
    return(train_data, valid_data, trainloader, validloader)


# Type 2: ImageFolder with no train/valid split, no weights
# make train/valid_datasets everything (with appropriate transforms), and then split in the sampler
def build_loader(data_dir, train_transforms, valid_transforms, valid_ratio, batchsize):
    
    train_dataset = datasets.ImageFolder(data_dir, transform = train_transforms)
    valid_dataset = datasets.ImageFolder(data_dir, transform = valid_transforms)
    
    valid_indices = random.sample(list(all_dataset.df.index),int(split_ratio*num_images))
    train_indices = [idx for idx in all_dataset.df.index if idx not in valid_indices]

    train_sampler = torch.utils.data.sampler.RandomSubsetSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.RandomSubsetSampler(valid_indices)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize, sampler = train_sampler)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batchsize, sampler = valid_sampler)


# Type 3: Folder of images
# new dataset class: builds dataset on indices, removes images not in dataset, can filter other features
# build_train_valid_loaders(): splits dataset into train/valid_dataset

class ImageDataSet(torch.utils.data.Dataset):

    def __init__(self,csv_path,images_path,these_indices=None,transforms=None):
        """
        Input:
            csv_path: path to csv file with metadata
            images_path: path to images
            these_indices: list of indices to keep in this dataset
            transform: transforms to be applied
        """
        self.df = pd.read_csv(csv_path)
        self.df.set_index("filename", inplace = True)
        self.images_path = images_path
        self.these_indices = these_indices
        self.transforms = transforms
        
        # remove rows that don't include the appropriate filenames:
        if self.these_indices:
            self.df.drop([idx for idx in self.df.index if idx not in these_indices], inplace=True)
            
        # Data Processing:
        # if view not PA, drop the row
        self.df.drop(self.df[self.df.view != 'PA'].index, inplace=True)
        # if image DNE, drop the row
        self.df.drop([idx for idx  in self.df.index if self.df.filename[idx] not in os.listdir(images_path)], inplace=True)
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            'Generates one sample of data'
            if torch.is_tensor(idx):
            idx = idx.tolist()
            
            image_path = images_path+"/"+self.df['filename'].iloc[idx]
            image = Image.open(image_path).convert('RGB')
            if self.transforms:
                image = self.transforms(image)
                
            if self.df['finding'].iloc[idx] != 'COVID-19':
                finding = 0
            else:
                finding = 1
                    
            return image, finding


def build_train_valid_loaders(csv_path,images_path,train_transform,valid_transforms,split_ratio,batchsize):
        """
    Input:
        - csv_path: path to csv file with metadata
        - images_path: path to images
        - train/valid transform: transforms to be applied to training, validation data
        - split_ratio: float between 0,1 indicating percentage of data to split into validation
        - batchsize = batch size
    """

    all_dataset = ImageDataSet(csv_path,images_path)
    num_images = len(all_dataset)
    # all_dataset already does some data processing: removes images not in data and images not PA

    
    # build train, valid datasets
    valid_indices = random.sample(list(all_dataset.df.index),int(split_ratio*num_images))
    train_indices = [idx for idx in all_dataset.df.index if idx not in valid_indices]
    valid_dataset = ImageDataSet(csv_path,images_path,valid_indices,valid_transforms)
    train_dataset = ImageDataSet(csv_path,images_path,train_indices,train_transforms)

    # build weights
    train_covid = len(train_dataset.df[train_dataset.df['finding'] == 'COVID-19'])
    train_class_weights = [train_covid / (len(train_dataset) - train_covid), (len(train_dataset) - train_covid) / train_covid]
    train_weights = [train_class_weights[finding] for image,finding in train_dataset]
    
    valid_covid = len(valid_dataset.df[valid_dataset.df['finding'] == 'COVID-19'])
    valid_class_weights = [valid_covid / (len(valid_dataset) - valid_covid), (len(valid_dataset) - valid_covid) / valid_covid]
    valid_weights = [valid_class_weights[finding] for image,finding in valid_dataset]

    # build samplers and loaders
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_dataset))
    valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(valid_weights, len(valid_dataset))
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize, sampler = train_sampler)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batchsize, sampler = valid_sampler)
    
    print("Data loaded!")
    return trainloader, validloader
                                                                                            
## can apply weights to the above in two different ways:
## first: if there are only two classes, and the class is given by a value of a dataframe
# (manipulating dataframes is in general much faster than iterating through the whole dataset):
train_covid = len(train_dataset.df[train_dataset.df['finding'] == 'COVID-19'])
train_class_weights = [train_covid / (len(train_dataset) - train_covid), (len(train_dataset) - train_covid) / train_covid]
train_weights = [train_class_weights[finding] for image,finding in train_dataset]
    
valid_covid = len(valid_dataset.df[valid_dataset.df['finding'] == 'COVID-19'])
valid_class_weights = [valid_covid / (len(valid_dataset) - valid_covid), (len(valid_dataset) - valid_covid) / valid_covid]
valid_weights = [valid_class_weights[finding] for image,finding in valid_dataset]
    
## second: iterate through the samples of the dataset# build weights
class_counts = {}
class_counts['train'] = dict(Counter(sample[1] for sample in train_dataset))
class_counts['valid'] = dict(Counter(sample[1] for sample in valid_dataset))

train_weights = [class_counts['train'][sample[1]] for sample in train_dataset]
valid_weights = [class_counts['valid'][sample[1]] for sample in valid_dataset]


## in both cases, building the sampler and loaders remains the same:
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_dataset))
valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(valid_weights, len(valid_dataset))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize, sampler = train_sampler)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batchsize, sampler = valid_sampler)
