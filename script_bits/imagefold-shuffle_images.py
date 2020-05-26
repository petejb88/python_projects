# ---------- Shuffle Images ----------
#
# Given a folder of images, creates directory of symlinks (with train/valid subfolders) to be interpreted by ImageFolder
#
# shuffle_images():
# --- delete/make_cat_folders(): deletes old and creates new train/valid folders, with subfolders given by classes
# --- list_images_in_data(): produces a list of images which are in the dataset
# --- randomly partitions images_in_data into trian/valid
# --- for each image in images_in_data, creates symlink in opporpriate train/valid and class
# --- test(): tests that each image is in the correct class folder, counts number of train/valid images, and compares against expected



def shuffle_images(data,images_path,p,column='finding'):
    '''
    Shuffle the images (as symbolic links) into training and validation folders.
    
    Inputs: 
        - p = percentage split into validation
        - data = metadata
        - images_path = path to images
        - column = column of dataset to consider as classes/categories
        
    Warning: unless metadata is perfect (no missing image files and no missing image rows), there will be some variability in the number of validation images
    '''
    
    classes = data[column].unique()
    delete_cat_folders(images_path,['train', 'valid'])
    make_cat_folders(images_path,['train', 'valid'], names,name_to_cat)
    
    images, images_in_data, data_not_in_images = list_images_in_data(data,images_path)
    
    valid_nums = random.sample(range(len(images)),int(len(images_in_data)*p))
    for image in images_in_data:
        image_data = data[data['filename']==image]
        image_index = image_data.index[0]
        image_column = image_data[column][image_index]
        image_cat = name_to_cat[image_column]
        if image_index in valid_nums:
            os.symlink(images_path+"/"+image, images_path+"/valid/"+image_cat+"/"+image)
        else:
            os.symlink(images_path+"/"+image, images_path+"/train/"+image_cat+"/"+image)
            
    test(data,p,images_in_data,column,name_to_cat,['train','valid'])


    
# utility functions

def delete_cat_folders(images_path,new_folders = ['train', 'valid']):
    for folder in new_folders:
        try:
            shutil.rmtree(images_path+"/"+folder)
            print("Folder {} deleted".format(folder))
        except:
            print("Folder {} doesn't exist!".format(folder))
        
def make_cat_folders(images_path,classes,new_folders = ['train', 'valid']):
    for folder in new_folders:
        try:
            os.mkdir(images_path+"/"+folder)
            print("Folder {} created".format(folder))
        except:
            print('Folder {} already exists!'.format(folder))
        for cls in classes:
            try:
                os.mkdir(images_path+"/"+folder+"/"+cls)
                print("Folder {}/{} created".format(folder,cls))
            except:
                print("Folder {}/{} already exists!".format(folder,cls))

def list_images_in_data(data,images_path):
    images = [image for image in os.listdir(images_path) if os.path.isfile(images_path+"/"+image)]
    images_not_in_data = []
    data_not_in_images = []
    for image in images:
        image_data = data[data['filename']==image]
        if len(image_data) == 0:
            images_not_in_data.append(image)

    images_in_data = [image for image in images if (image not in images_not_in_data)]
    for filename in data['filename']:
        if filename not in os.listdir(images_path):
            data_not_in_images.append(filename)
    print("How many in folder: {}, How many not in data: {}, How many in data: {}, How many data not in images: {}".format(
                len(images),
            len(images_not_in_data),
            len(images_in_data),
                len(data_not_in_images)))
    return images, images_in_data, data_not_in_images
        
        

def test_folder(data,p,images_in_data,column,name_to_cat,folder):
    count = 0
    wrong_cat = []
    for subfolder in os.listdir(images_path+"/"+folder):
        for image in os.listdir(images_path+"/"+folder+"/"+subfolder):
            count += 1
            image_data = data[data['filename']==image]
            image_index = image_data.index[0]
            image_column = image_data[column][image_index]
            image_cat = name_to_cat[image_column]
            if image_cat != subfolder:
                wrong_cat.append(image)
    return count, wrong_cat


def test(data,p,images_in_data,column,name_to_cat,new_folders):
    count = []
    wrong_cat = []
    for i,folder in enumerate(new_folders):
        count.append(test_folder(data,p,images_in_data,column,name_to_cat,folder)[0])
        wrong_cat.append(test_folder(data,p,images_in_data,column,name_to_cat,folder)[1])

    print("Valid Count: {}, Valid Expected: {}".format(count[1], int((p*len(images_in_data)))))
    print("Train Count: {}, Train Expected: {}".format(count[0], int((1-p)*len(images_in_data))))
    print("Total Count: {}, Total Expected: {}".format(sum(count), len(images_in_data)))
    print("Valid Wrong Cat: {}".format(wrong_cat[1]))
    print("Train Wrong Cat: {}".format(wrong_cat[0]))

 # outdated functions

def make_cat_dicts(images_path,names):
    cat_to_name = {}
    name_to_cat = {}
    names.sort()
    for i,name in enumerate(names):
        cat_to_name[str(i)] = name
        name_to_cat[name] = str(i)
    
    json_object = json.dumps(cat_to_name, indent = 4)
    with open(data_path+"/cat_to_name.json", "w") as outfile:
        outfile.write(json_object)
        
    print("cat_to_name, name_to_cat created!")
    print(name_to_cat)
    return cat_to_name, name_to_cat
                                                                             
