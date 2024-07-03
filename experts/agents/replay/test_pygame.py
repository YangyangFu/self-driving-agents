import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the display
screen_width, screen_height = 1920, 1080
screen = pygame.display.set_mode((screen_width, screen_height))

# Assuming you have three image arrays: image1, image2, image3
# Each with shape (w, h, 3)
# load image array from png file
image1_surf = pygame.image.load('front.png')
image2_surf = pygame.image.load('left.png')
image3_surf = pygame.image.load('right.png')
image1 = pygame.surfarray.array3d(image1_surf)
image2 = pygame.surfarray.array3d(image2_surf)
image3 = pygame.surfarray.array3d(image3_surf)

def array_to_surface(arr):
    # Convert numpy array to Pygame surface
    arr = arr.astype(np.uint8)
    return pygame.surfarray.make_surface(arr)

# Convert image arrays to Pygame surfaces
surface1 = array_to_surface(image1)
surface2 = array_to_surface(image2)
surface3 = array_to_surface(image3)
print(surface1.get_width(), surface1.get_height())
# Calculate the width for each image (1/3 of the screen width)
image_width = screen_width // 3

# Scale images to fit 1/3 of the screen width while maintaining aspect ratio
def scale_image(surface, target_width):
    aspect_ratio = surface.get_width() / surface.get_height()
    new_height = int(target_width / aspect_ratio)
    return pygame.transform.scale(surface, (target_width, new_height))

scaled_surface1 = scale_image(surface1, image_width)
scaled_surface2 = scale_image(surface2, image_width)
scaled_surface3 = scale_image(surface3, image_width)
print(scaled_surface1.get_width(), scaled_surface1.get_height())

# Calculate positions for the images
pos1 = (0, (screen_height - scaled_surface1.get_height()) // 2)
pos2 = (image_width, (screen_height - scaled_surface2.get_height()) // 2)
pos3 = (2 * image_width, (screen_height - scaled_surface3.get_height()) // 2)


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))  # Fill with black

    # Blit the images onto the screen
    screen.blit(scaled_surface1, pos1)
    screen.blit(scaled_surface2, pos2)
    screen.blit(scaled_surface3, pos3)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()