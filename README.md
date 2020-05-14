# transfer learning image classifier

train.py:
- Use transfer learning, based on one of the Torchvision pretrained models, to produce a model trained on given data to classify images
- input data: train/ and valid/ with images in subfolders named by an appropriate (numerical) class
- saves model as checkpoint

predict.py:
- loads checkpoint to (re)create model, trained as above
- predicts class for given image file path

example_data: 
- example data_dir setup
- includes cat_to_name.json: dictionary of categorical output classes from data_dir with their associated English identifications

