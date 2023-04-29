import torch 
import torch.nn as nn 
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils import data 




def clip_gradient(model, norm = 2.0):
	"""Rescales norm of computed gradients.

	Parameters
	----------
	model: nn.Module 
		Module.

	clip: float
		Maximum norm.

	"""

	for p in model.parameters():
		if p.grad is not None:
			param_norm = p.grad.data.norm()
			clip_coeff = clip / (param_norm + 1e-6)
			if clip_coef < 1:
				p.grad.data.mul_(clip_coef)





class MultiCropWrapper(nn.Module):
    """A class for the forward pass of all the multiple crops.
    This class assumes that all the crops are of the same shape.

    Parameters:
    -----------
    backbone: `torch.nn.Module`
            The Vision Transformer 
    """


    def __init__(self, backbone, head):
        super().__init__()
        backbone.head = nn.Identity()
        self.backbone = backbone 
        self.head = head

    def forward(self, x):
        """Run the forward pass 

        All the crops are concatenated along the batch dimension 
        and then a single forward pass is done. The final result 
        is then chuncked back to per crop tensors. 

        Parameters:
        -----------
        x: list 
           list of 'torch.Tensor' each of shape (n_samples, 3, size, size).

        Returns
        -------
        tuple 
            Tuple of 'torch.Tensor' each of shape '(n_samples, out_dim)'
        """

        n_crops = len(x)
        concatenated = torch.cat(x, dim = 0) # (n_samples * n_crops, in_dim)
        cls_token = self.backbone(concatenated) #(n_samples * n_crops, out_dim)
        logits = self.head(cls_token) #(n_samples * n_crops, out_dim)
        chunks = logits.chunk(n_crops) #List: n_crops, (n_samples, out_dim)

        return chunks 




class DataAugmentation(object):


    def __init__(self, global_crops_scale = (0.4, 1), local_crops_scale = (0.1, 0.4), n_local_crops = 8, size = 32):
        self.n_local_crops = n_local_crops

        RandomGaussianBlur = lambda p: transforms.RandomApply([transforms.GaussianBlur(kernel_size = 5, sigma = (0.1, 2))], p = p)

        flip_and_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5), #0.5
            transforms.RandomApply([
                transforms.ColorJitter(brightness = 0.4, contrast = 0.4, saturation=0.2, hue=.1)], p = 0.8), #0.8

            transforms.RandomGrayscale(p = 0.2) 
            ])


        normalize = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])


        self.global1 = transforms.Compose([
            #transforms.Resize(size = 224),                                  
            transforms.RandomResizedCrop(size, scale = global_crops_scale, interpolation = transforms.InterpolationMode.BICUBIC),  
            flip_and_jitter, 
            RandomGaussianBlur(p = 0.1),
            normalize
            ])

        #Differnce w/ global1: 1) Here p global is much lower 2) Here we use solarization. 
        self.global2 = transforms.Compose([
            #transforms.Resize(size = 224),                                    
            transforms.RandomResizedCrop(size, scale = global_crops_scale, interpolation =  transforms.InterpolationMode.BICUBIC),
            flip_and_jitter, 
            RandomGaussianBlur(p = 0.1),
            transforms.RandomSolarize(170, p = 0.2),
            normalize
            ])


        self.local =  transforms.Compose([
            #transforms.Resize(size = 224),                                   
            transforms.RandomResizedCrop(size, scale = local_crops_scale, interpolation = transforms.InterpolationMode.BICUBIC),
            flip_and_jitter, 
            RandomGaussianBlur(0.1),
            normalize
            ])	



    def __call__(self, img):
        """ Apply transformation.

        Parameters
        ----------
        img: PIL.Image
            input image

        Returns
        -------
        all_crops: list
            list of 'torch.Tensor' representing different views of the input 'img'		
        """

        all_crops = []
        all_crops.append(self.global1(img))
        all_crops.append(self.global2(img))

        all_crops.extend([self.local(img) for _ in range(self.n_local_crops)])

        return all_crops







#============= Reading an image====================
def read_path():
    dir1 = os.path.dirname(__file__)
    path = "image/img.PNG"
    img_path = os.path.join(dir1, path)

    return img_path


def read_image(path):
    '''returns array of an image and the image itself'''

    img = Image.open(path)
    print(img.format) # prints format of image
    print(img.mode)  # prints mode of image
    print(img.size)

    #img.show()  

    return img



def display_images(imgs, n_images, rows, columns):
    """A function that displays the augmented images of DINO.

    Parameters
    ----------
    imgs: <class torchvision.datasets> OR <torch.Tensor>
        1- The shape of the input is roughly: (n_elements,) where each element 
        is a tuple with length equal to the number of crops(10 crops for example).
        In Dino we have 2 global crops and 8 local crops. Each element in this 
        tuple is an augmented image with size `(3, dim, dim)`.

        2- If the input is an instance of `torch.Tensor`, the first if condition will 
        be executed.

    n_images: int
            The number of images to display.

    rows: int 
        The number of rows of the final plot.

    columns: int
            The number of columns of the final plot.

    Returns
    -------
    It displays the images. 
    """

    fig = plt.figure(figsize = (12, 15))
    fig.subplots_adjust(hspace = .05, wspace = 0.05)

    if isinstance(imgs, torch.Tensor):
        for i in range(n_images):
            ax = fig.add_subplot(2, 5, i+1, xticks = [], yticks = [])
            ax.imshow(imgs[i].permute(1, 2, 0))

        plt.show()

    else:
        pos = 1
        for i in range(n_images):
            for sub_img in imgs[i][0]:
                ax = fig.add_subplot(rows, columns, pos, xticks= [], yticks = [])
                ax.imshow(sub_img.permute(2, 1, 0))
                pos += 1 

        plt.show()









if __name__ == '__main__':
	
	path = r'D:\ML\KNTU_Courses\datasets\Cifar 10\Cifar10 dino'
	augment = DataAugmentation()

	data_train = datasets.CIFAR10(path, train = True, download = True, transform = augment)
	data_test = datasets.CIFAR10(path, train = False, download = True)


	train_loader = data.DataLoader(data_train, batch_size = 64, shuffle = True, drop_last = True)

	sample = next(iter(train_loader))

	for idx, (images, labels) in enumerate(train_loader):
		if idx == 0:
			
			# global1 = images[0]
			# global2 = images[1]

			# local1 = images[3]
			# local2 = images[4]
			# local3 = images[5]
			# local4 = images[6]


			# display_images(global1, 10, 5, 2)
			# display_images(global2, 10, 5, 2)
			# display_images(local1, 10, 5, 2)
			# display_images(local2, 10, 5, 2)
			# display_images(local3, 10, 5, 2)
			# display_images(local4, 10, 5, 2)
			print(len(images))
			print(type(labels))
			print(len(labels))

			break







