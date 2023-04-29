import torch 
import torch.nn as nn 
import numpy 
from dino import DinoLoss, Dino
from utils import DataAugmentation, MultiCropWrapper
from torchvision import datasets, transforms  
import matplotlib.pyplot as plt  
from torch.utils import data
from utils import clip_gradient


def train_func(train_loader, student, teacher, optimizer, loss_func, momentum_teacher, max_epochs = 100,  
                validation_loader = None, batch_size = 128, scheduler = None, device = None, test_loader = None, 
                train_loader_plain = None, clip_grad = None):

    """Train function for dino. It takes two identical models, the teacher and student, 
    and only the student model is trained. Note that Both the teacher and the student
    model share the same architecture, and initially, they also have the same parameters.
    The parameters of the teacher are updated using the exponential moving average of 
    the student model.


    Parameters
    ----------
    train_loader: Instance of `torch.utils.data.DataLoader`

    student: Instance of `torch.nn.Module'
            The Vision Transformer as the student model.

    teacher: Instance of `torch.nn.Module'
        The Vision Transformer as the teacher model. 

    optimizer: Instance of `torch.optim`
            Optimizer for training.

    loss_func: Instance of `torch.nn.Module'
            Loss function for the training.


    Returns
    -------
    history: dict 
            Returns training and validation loss. 

    """

    n_batches_train = len(train_loader)
    n_batches_val = len(validation_loader)
    n_samples_train = batch_size * n_batches_train
    n_samples_val = batch_size * n_batches_val


    losses = []
    accuracy = []
    validation_loss = []
    validation_accuracy = []


    for epoch in range(max_epochs):
        running_loss, correct = 0, 0
        for images, labels in train_loader:
            if device:
                images = [img.to(device) for img in images]
                labels = labels.to(device)

            #Training
            student.train()
            student.training = True
            cls_student = student(images)
            cls_teacher = teacher(images[:2]) #Teacher only gets the global crops
            loss = loss_func(student_output = cls_student, teacher_output = cls_teacher)

            running_loss += loss.item()


            #================= BACKWARD AND OPTIMZIE  ====================================   
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(student, clip_grad)
            optimizer.step()

            		
            #================== Updating the teacher's parameters ========================
            with torch.no_grad():
                for student_params, teacher_params in zip(student.parameters(), teacher.parameters()):
                    teacher_params.data.mul_(momentum_teacher)
                    teacher_params.data.add_((1 - momentum_teacher) * student_params.detach().data)


        loss_epoch = running_loss / n_batches_train
        losses.append(loss_epoch)
        scheduler.step()

        print('Epoch [{}/{}], Training Loss: {:.4f}'
            .format(epoch + 1, max_epochs, loss_epoch), end = '  ')


        #====================== Validation ============================
        if validation_loader:
            student.eval()   

            acc_logistic, acc_val_logistic, acc_val_knn = compute_knn(student.backbone, train_loader_plain, validation_loader)
            validation_accuracy.append(acc_val_knn)

            print('Training accuracy Logistic [{:.3f}], Validation accuracy Logistic [{:.3f}], Validation KNN [{:.3f}]'
                .format(acc_logistic, acc_val_logistic, acc_val_knn))
        
        #====================== Saving the Model ============================  
        model_save_name = 'dino.pt'
        path = F"/kaggle/working/{model_save_name}"
        torch.save(student.state_dict(), path)
    
    #====================== Testing ============================      
    if test_loader:
        correct = 0
        total = 0

        for images, labels in test_loader:
            if device:
                images = images.to(device)
                labels = labels.to(device)

            n_data = images[0]
            total += n_data
            outputs = student(images)
            predictions = outputs.argmax(1)
            correct += int(sum(predictions == labels))

        accuracy = correct / total 
        print('Test Accuracy: {}'.format(accuracy))



    return {'loss': losses, 'accuracy': accuracy, 
            'val_loss': validation_loss, 'val_accuracy': validation_accuracy}



def main(parameters):

    #=============================Preparing Data==================================
    path = F"/content/gdrive/My Drive/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plain_augmentation = transforms.Compose([
        #transforms.Resize(32),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    dino_augmentation = DataAugmentation(n_local_crops = parameters['n_crops'] - 2)
    dataset_train = datasets.CIFAR10(path, download = True, train = True, transform = dino_augmentation)
    dataset_test = datasets.CIFAR10(path, download = True, train = False, transform = plain_augmentation)
    dataset_train_evaluation = datasets.CIFAR10(path, download = True, train = True, transform = plain_augmentation)
    dataset_validation, dataset_test = torch.utils.data.random_split(dataset_test, [8000, 2000])


    train_loader = data.DataLoader(dataset_train, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)
    val_loader = data.DataLoader(dataset_validation, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)
    train_loader_plain = data.DataLoader(dataset_train_evaluation, batch_size = parameters['batch_size'], drop_last = True, num_workers = 2)


    #=============================Preparing The Model==================================
    student = Dino(img_size = parameters['img_size'], patch_size = parameters['patch_size'], 
        n_classes = parameters['n_classes'], embed_dim = parameters['embed_dim'], layers = parameters['layers'], 
        n_heads = parameters['n_heads'], early_cnn = parameters['early_cnn'])

    teacher = Dino(img_size = parameters['img_size'], patch_size = parameters['patch_size'], 
        n_classes = parameters['n_classes'], embed_dim = parameters['embed_dim'], layers = parameters['layers'], 
        n_heads = parameters['n_heads'], early_cnn = parameters['early_cnn'])
    
    head_student = DinoHead(in_dim = 384, hidden_dim = 384, out_dim = parameters['out_dim'], n_layers = 3, norm_last_layer = True)
    head_teacher = DinoHead(in_dim = 384, hidden_dim = 384, out_dim = parameters['out_dim'], n_layers = 3, norm_last_layer = True)
    student = MultiCropWrapper(student, head_student)
    teacher = MultiCropWrapper(teacher, head_teacher)
    student, teacher = student.to(device), teacher.to(device)

    teacher.load_state_dict(student.state_dict()) #Making sure that the two networks' parameters are the same

    for params in teacher.parameters(): 
        params.requires_grad = False

    criterion = DinoLoss(parameters['out_dim'], teacher_temp = parameters['teacher_temp'], 
        student_temp = parameters['student_temp'], center_momentum = parameters['center_momentum']).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr = parameters['lr'], weight_decay = parameters['weight_decay'])
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 100, eta_min = 1e-6)
    scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=5, after_scheduler = base_scheduler)
     
    momentum_teacher = parameters['momentum_teacher']
    history, modell = train_func(train_loader = train_loader, student = student, teacher = teacher, 
        optimizer = optimizer, loss_func = criterion, validation_loader = val_loader, 
        device = device, scheduler = scheduler, batch_size = parameters['batch_size'], 
        max_epochs = parameters['max_epochs'], momentum_teacher = momentum_teacher, 
        train_loader_plain = train_loader_plain, clip_grad = parameters['clip_grad'])

    return student, history



if __name__ == '__main__':

    parameters = {'batch_size': 512, 'lr': 7.5e-6, 'weight_decay': 0.05, 'img_size': 32, 'n_crops': 4, 
                 'layers' : 6, 'n_heads' : 6, 'patch_size' : 8,  'early_cnn' : True, 'n_classes' : 384, 
                 'embed_dim' : 384, 'out_dim': 1024, 'teacher_temp' : 0.04, 'student_temp' : 0.1, 
                 'center_momentum' : 0.996, 'max_epochs' : 100, 'momentum_teacher': 0.9, 'clip_grad': 2.0}

    model, history = main(parameters)
    
    
    
	#=============================Validation & Visualizing Embeddings ==================================
 	





