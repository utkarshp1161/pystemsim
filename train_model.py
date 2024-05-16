from package_files.utils import *
from package_files.networks import *
from package_files.loss_func import GombinatorialLoss

# Create dataloaders
images_dir='./MoS2gb_dataset/images'
labels_dir='./MoS2gb_dataset/labels'
train_loader, val_loader, test_loader = get_dataloaders(images_dir, labels_dir, batch_size = 4, val_split=0.2, test_split=0.1)

# Model Params
input_channels = 1
num_classes = 6    # number of output classes
num_filters = [16, 32, 64, 128, 256]

# Check if a GPU is available and if not, use a CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps') # backend for Apple silicon GPUs
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Create and train model with CombinatorialGroupLoss
model = Unet(input_channels, num_classes, num_filters, dropout = 0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss = GombinatorialLoss(group_size = num_classes//2, loss = 'Dice', epsilon=1e-6, class_weights = None, alpha=3)
                         
model, train_loss, val_loss = train_model(model, train_loader, val_loader, n_epochs = 100,
                                          criterion = loss, optimizer = optimizer, device = device,
                                          save_name = './trained_models/GL_norotation_v3_similarity_and_falsepositive_penalty_.pth')

# # Create and train model with CE loss
# model = Unet(input_channels, num_classes, num_filters)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# loss = torch.nn.CrossEntropyLoss()
# model, train_loss, val_loss = train_model(model, train_loader, val_loader, n_epochs = 10000,
#                                           criterion = loss, optimizer = optimizer, device = device,
#                                           save_name = './trained_models/MoS2gb_model_CELoss_K3P1.pth')
