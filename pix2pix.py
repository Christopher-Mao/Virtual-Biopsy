import os
import glob
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

import torch
import torch.nn as nn

if torch.cuda.is_available():
  device = torch.device("cuda")
  print("GPU is available.")
else:
  device = torch.device("cpu")
  print("GPU is not available.")

print(f"Current device: {device}")

base_path = '/work/zbq615/Data'

# Define the custom Dataset class
class OCTHistologyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self):
        pairs = []
        # Loop through each directory in the root directory
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            oct_dir = os.path.join(folder_path, 'OCT')
            he_dir = os.path.join(folder_path, 'H&E')

            # Match OCT and H&E images by filenames
            for oct_image in glob.glob(os.path.join(oct_dir, '*_OCT.png')):
                base_name = os.path.basename(oct_image).replace('_OCT.png', '')
                he_image = os.path.join(he_dir, f"{base_name}_hist.jpg")

                if os.path.exists(he_image):
                    pairs.append((oct_image, he_image))
        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        oct_path, he_path = self.image_pairs[idx]

        # Load images
        oct_image = Image.open(oct_path).convert('RGB')
        he_image = Image.open(he_path).convert('RGB')

        # Apply transformations
        if self.transform:
            oct_image = self.transform(oct_image)
            he_image = self.transform(he_image)

        return {'input': oct_image, 'output': he_image}

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match pix2pix input
    transforms.ToTensor(),
])

# Create the dataset
dataset = OCTHistologyDataset(root_dir=base_path, transform=transform)

# Split the dataset into train, val, and test sets (e.g., 70%, 15%, 15%)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
batch_size = 8
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Example usage: iterate through one batch
for batch in train_loader:
    inputs = batch['input']  # OCT images
    targets = batch['output']  # H&E images
    print(inputs.shape, targets.shape)
    break

# Define an encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# Define a decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
        super(DecoderBlock, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.insert(2, nn.Dropout(0.5))  # Apply dropout after BatchNorm
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Define the generator using a U-Net structure
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.encoder1 = EncoderBlock(in_channels, features)            # 64
        self.encoder2 = EncoderBlock(features, features * 2)           # 128
        self.encoder3 = EncoderBlock(features * 2, features * 4)       # 256
        self.encoder4 = EncoderBlock(features * 4, features * 8)       # 512
        self.encoder5 = EncoderBlock(features * 8, features * 8)       # 512
        self.encoder6 = EncoderBlock(features * 8, features * 8)       # 512
        self.encoder7 = EncoderBlock(features * 8, features * 8)       # 512
        self.bottleneck = EncoderBlock(features * 8, features * 8)     # 512

        # Decoder
        self.decoder1 = DecoderBlock(features * 8, features * 8, dropout=True)  # 512
        self.decoder2 = DecoderBlock(features * 16, features * 8, dropout=True) # 512
        self.decoder3 = DecoderBlock(features * 16, features * 8, dropout=True) # 512
        self.decoder4 = DecoderBlock(features * 16, features * 8)               # 512
        self.decoder5 = DecoderBlock(features * 16, features * 4)               # 256
        self.decoder6 = DecoderBlock(features * 8, features * 2)                # 128
        self.decoder7 = DecoderBlock(features * 4, features)                    # 64
        self.final = nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        bottleneck = self.bottleneck(e7)

        # Decoding path with skip connections
        d1 = self.decoder1(bottleneck)
        d2 = self.decoder2(torch.cat([d1, e7], dim=1))
        d3 = self.decoder3(torch.cat([d2, e6], dim=1))
        d4 = self.decoder4(torch.cat([d3, e5], dim=1))
        d5 = self.decoder5(torch.cat([d4, e4], dim=1))
        d6 = self.decoder6(torch.cat([d5, e3], dim=1))
        d7 = self.decoder7(torch.cat([d6, e2], dim=1))
        output = self.final(torch.cat([d7, e1], dim=1))

        return torch.tanh(output)  # Apply tanh to scale output to [-1, 1]

def generator_loss_fn(disc_fake_output, gen_output, target, lambda_l1=100):
    # GAN Loss (Adversarial Loss)
    # Create a tensor of ones to represent "real" labels
    real_labels = torch.ones_like(disc_fake_output)
    gan_loss = nn.BCEWithLogitsLoss()(disc_fake_output, real_labels)

    # L1 Loss (Mean Absolute Error)
    l1_loss = nn.L1Loss()(gen_output, target)

    # Total Generator Loss
    total_gen_loss = gan_loss + lambda_l1 * l1_loss
    return total_gen_loss


# Define a discriminator block
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DiscriminatorBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# Define the PatchGAN discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(PatchGANDiscriminator, self).__init__()

        # Input channels are doubled to include both input and target/generated images
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Discriminator layers
        self.block1 = DiscriminatorBlock(features, features * 2)      # 128
        self.block2 = DiscriminatorBlock(features * 2, features * 4)  # 256
        self.block3 = DiscriminatorBlock(features * 4, features * 8)  # 512

        # Final layer (no BatchNorm)
        self.final = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, input_image, target_image):
        # Concatenate the input and target/generated images along the channel dimension
        x = torch.cat([input_image, target_image], dim=1)

        # Forward pass through the network
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final(x)

        # Output is (batch_size, 1, 30, 30)
        return x

# Define the discriminator loss function
def discriminator_loss_fn(disc_real_output, disc_fake_output):
    # Create labels
    real_labels = torch.ones_like(disc_real_output)
    fake_labels = torch.zeros_like(disc_fake_output)

    # Real loss: real images classified as real
    real_loss = nn.BCEWithLogitsLoss()(disc_real_output, real_labels)

    # Generated loss: generated images classified as fake
    generated_loss = nn.BCEWithLogitsLoss()(disc_fake_output, fake_labels)

    # Total Discriminator Loss
    total_loss = real_loss + generated_loss
    return total_loss

# Initialize the generator and discriminator
generator = UNetGenerator(in_channels=3, out_channels=3)  # Assuming RGB images for both input and output
discriminator = PatchGANDiscriminator(in_channels=3)      # Also assuming RGB input and output

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)


import torch.optim as optim

# Initialize optimizers for both generator and discriminator
generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))


import os

# Function to save checkpoints
def save_checkpoint(generator, discriminator, g_optimizer, d_optimizer, epoch, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    # Save the generator and discriminator state_dicts, optimizers, and epoch
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'epoch': epoch
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Function to load checkpoints
def load_checkpoint(generator, discriminator, g_optimizer, d_optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    # Load the saved state dictionaries
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

    # Return the saved epoch for resuming training
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
    return epoch


import matplotlib.pyplot as plt

# Function to plot results from the generator
def plot_predictions(generator, test_loader, device, num_images=3, output_dir="pix2pix_training_output"):
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()  # Set generator to evaluation mode
    
    with torch.no_grad():  # No gradient tracking needed for inference
        # Get a batch of test images
        for batch in test_loader:
            # Assuming test_loader returns a tuple (input_image, target_image)
            input_images = batch['input'].to(device)
            target_images = batch['output'].to(device)

            # Generate output images
            generated_images = generator(input_images)

            # Plot the results
            plt.figure(figsize=(15, num_images * 5))
            for i in range(num_images):
                # Display input image
                plt.subplot(num_images, 3, 3 * i + 1)
                plt.imshow(input_images[i].cpu().permute(1, 2, 0))
                plt.title("Input Image")
                plt.axis("off")

                # Display target image
                plt.subplot(num_images, 3, 3 * i + 2)
                plt.imshow(target_images[i].cpu().permute(1, 2, 0))
                plt.title("Target Image")
                plt.axis("off")

                # Display generated image
                plt.subplot(num_images, 3, 3 * i + 3)
                plt.imshow(generated_images[i].cpu().permute(1, 2, 0))
                plt.title("Generated Image")
                plt.axis("off")

            # Save the plot
            output_path = os.path.join(output_dir, f"epoch_{epoch + 1}.png")
            plt.savefig(output_path)
            print(f"Saved predictions for epoch {epoch + 1} to {output_path}")
            plt.close()
            break  # Display only the first batch

    generator.train()  # Set generator back to training mode


from tqdm import tqdm

# Define the number of epochs
num_epochs = 200

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    generator.train()  # Set generator to training mode
    discriminator.train()  # Set discriminator to training mode

    # Track losses for each epoch
    gen_loss_epoch = 0
    disc_loss_epoch = 0

    for batch in tqdm(train_loader):
        # Load the input and target images from the batch
        '''
        input_images, target_images = batch
        input_images = input_images.to(device)
        target_images = target_images.to(device)
        '''
        input_images = batch['input'].to(device)
        target_images = batch['output'].to(device)

        ## ---- Train Discriminator ---- ##
        # Generate fake images using the generator
        generated_images = generator(input_images)

        # Discriminator's predictions on real and generated images
        disc_real_output = discriminator(input_images, target_images)
        disc_generated_output = discriminator(input_images, generated_images.detach())

        # Calculate discriminator loss
        disc_loss = discriminator_loss_fn(disc_real_output, disc_generated_output)

        # Update discriminator
        discriminator_optimizer.zero_grad()
        disc_loss.backward()
        discriminator_optimizer.step()

        ## ---- Train Generator ---- ##
        # Get new discriminator output for generator loss
        disc_generated_output = discriminator(input_images, generated_images)

        # Calculate generator loss
        gen_loss = generator_loss_fn(disc_generated_output, generated_images, target_images)

        # Update generator
        generator_optimizer.zero_grad()
        gen_loss.backward()
        generator_optimizer.step()

        # Accumulate losses for display
        gen_loss_epoch += gen_loss.item()
        disc_loss_epoch += disc_loss.item()

    # Average losses for the epoch
    avg_gen_loss = gen_loss_epoch / len(train_loader)
    avg_disc_loss = disc_loss_epoch / len(train_loader)

    # Print losses for the epoch
    print(f"Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")

    # Save predictions every few epochs
    if (epoch + 1) % 5 == 0:
        plot_predictions(generator, test_loader, device, num_images=8)

    # Save model checkpoints every few epochs
    if (epoch + 1) % 5 == 0:
        save_checkpoint(generator, discriminator, generator_optimizer, discriminator_optimizer, epoch + 1)

