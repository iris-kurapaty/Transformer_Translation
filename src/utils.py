import matplotlib.pyplot as plt
import numpy as np
import torch 
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def plot_loss_acc(train_losses, train_acc, test_losses, test_acc):
  t = [t_items.item() for t_items in train_losses]
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(t)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

  return

def mis_classified_images(model, device, test_loader):
  incorrect_examples = []
  incorrect_labels = []
  incorrect_pred = []
  model.eval()
  for data,target in test_loader:

    data , target = data.to(device), target.to(device)
    output = model(data) # shape = torch.Size([batch_size, 10])
    pred = output.argmax(dim=1, keepdim=True) #pred will be a 2d tensor of shape [batch_size,1]
    idxs_mask = ((pred == target.view_as(pred))==False).view(-1)
    if idxs_mask.numel(): #if index masks is non-empty append the correspoding data value in incorrect examples
      incorrect_examples.append(data[idxs_mask].squeeze().cpu().numpy()) #
      incorrect_labels.append(target[idxs_mask].cpu().numpy())# #the corresponding target to the misclassified image
      incorrect_pred.append(pred[idxs_mask].squeeze().cpu().numpy()) ##the corresponiding predicted class of the misclassified image

      return incorrect_examples, incorrect_labels,incorrect_pred

# Let's visualize some of the images
def num_imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg= np.transpose(npimg, (1, 2, 0))
    return npimg

def plot_images(incorrect_examples, incorrect_labels,incorrect_pred, num_images):

  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  for i in range(1, num_images+1):
    img = incorrect_examples[0][i-1]
    plt.subplot(4, 5, i)
    plt.imshow(num_imshow(torch.from_numpy(img)))
    plt.axis('off')
    plt.title("actual: %s\npredicted: %s" % (classes[incorrect_labels[0][i-1]], classes[incorrect_pred[0][i-1]]), fontsize=8)
    plt.subplots_adjust(top=5, bottom=3, left=1, right=2)
  return

def display_gradcam(model, incorrect_examples, num_images, inv_normalize, transperancy, targets=None):
  # target_layers = [model.layer_2[-1]]
  target_layers = [model.layer_3.res_block.conv2]
  cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
  grad_cam_images = []
  # f, axes = plt.subplots(nrows=4, ncols=5, left=1, right=2)
  for i in range(num_images):
    # ax = axes.flat[i]
    input_tensor = torch.from_numpy(incorrect_examples[0][i])
    input_tensor = input_tensor.unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    img = input_tensor.squeeze(0).to('cpu')
    img = inv_normalize(img)
    rgb_img = np.transpose(img, (1, 2, 0))
    rgb_img = rgb_img.numpy()
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transperancy)
    # plt.subplot(4, 5, i+1)
    # plt.imshow(visualization)
    # plt.axis('off')
    # plt.title("actual: %s\npredicted: %s" % (classes[incorrect_labels[0][i]], classes[incorrect_pred[0][i]]), fontsize=8)
    # plt.subplots_adjust(top=5, bottom=3, left=1, right=2)

    grad_cam_images.append(visualization)
  return grad_cam_images