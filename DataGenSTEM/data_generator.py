# Description: This file contains the functions to generate synthetic data for the neural network training.
# By Austin Houston
# Date: 02/28/2024
# Updated: 05/10/2024

import dask
import numpy as np
import random
import dask.array as da
import scipy.special as sp
from scipy.ndimage import zoom, gaussian_filter
from skimage.draw import disk
from ase import Atoms
from ase.neighborlist import NeighborList


def get_xtal_matrix(xtal, n_cells = (1,1,1), rotation = 0, n_vacancies = 10, phonon_sigma = 0.01, axis_extent = None):
    """
    Generates an imaging crystal structure by manipulating an ACE atoms object.

    Parameters:
    xtal (Atoms): An Atomic Simulation Environment (ASE) Atoms object representing the crystal structure.
    n_cells (tuple of int, optional): A tuple (nx, ny, nz) defining the replication of the unit cell in each direction. Defaults to (1,1,1).
    rotation (float, optional): The angle in degrees to rotate the crystal structure around the center of mass. Defaults to 0.
    n_vacancies (int, optional): Number of vacancies to introduce in the crystal structure. Defaults to 10.
    phonon_sigma (float, optional): Standard deviation for the positional noise to simulate thermal effects (phonons). Defaults to 0.01.
    axis_extent (tuple of float, optional): A tuple (xmin, xmax, ymin, ymax) defining the axis extents to limit the positions of the atoms. If None, no cutting is performed. Defaults to None.

    Returns:
    Atoms: The modified ASE Atoms object representing the transformed crystal structure.
    
    This function performs several operations on the input crystal structure:
    1. Replicates the unit cell.
    2. Applies rotation around the center of mass.
    3. Introduces thermal vibrations through rattling.
    4. Optionally cuts atoms outside the specified imaging region.
    5. Introduces vacancies.
    """
    xtal = xtal * n_cells
    positions = xtal.get_positions()

    # get atom index closest to COM
    com = np.mean(positions, axis=0)
    dists = np.linalg.norm(positions - com, axis=1)
    com_atom = np.argmin(dists)

    # rotate atoms around COM atom
    theta = rotation * np.pi / 180 # deg to rad
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    translated_coords = positions[:,:2] - np.array(positions[com_atom, :2])
    rotated_atom_pos = np.dot(translated_coords, rotation_matrix)
    rotated_atom_pos += np.array(positions[com_atom, :2])
    new_positions = positions.copy()
    new_positions[:,:2] = rotated_atom_pos
    xtal.set_positions(new_positions)

    # add positional noise with frozen phonons, normal distribution
    xtal.rattle(stdev = phonon_sigma)
    rattled_positions = xtal.get_positions()

    # optionally: cut atoms outside the edge of the image region
    if axis_extent is not None:
        xmin, xmax, ymin, ymax = axis_extent        
        ids_to_delete = np.where((rattled_positions[:,0] < xmin) | (rattled_positions[:,0] > xmax) | 
                                 (rattled_positions[:,1] < ymin) | (rattled_positions[:,1] > ymax))[0]
        del xtal[ids_to_delete]
        
    # remove vacancies
    if n_vacancies > 0:
        n_atoms = len(xtal.get_atomic_numbers())
        vacancies = np.random.choice(n_atoms, n_vacancies, replace=False)
        del xtal[vacancies]
    
    return xtal
    

def make_holes(atoms: Atoms, n_holes: int, hole_size: float) -> Atoms:
    """
    Create holes in an Atoms object by deleting atoms around randomly selected positions.

    Parameters:
    - atoms (ase.Atoms): The input Atoms object.
    - n_holes (int): The number of holes to create.
    - hole_size (float): The radius of each hole.

    Returns:
    - ase.Atoms: The modified Atoms object with holes.
    """
    # Step 1: Randomly select n_holes atoms
    num_atoms = len(atoms)
    selected_indices = random.sample(range(num_atoms), n_holes)

    # Step 2: Find and delete atoms within radius hole_size
    for index in selected_indices:
        # Get the position of the selected atom
        pos = atoms[index].position

        # Create a NeighborList to find atoms within hole_size
        cutoffs = [hole_size / 2] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        # Find atoms within hole_size around the selected atom
        indices, offsets = nl.get_neighbors(index)
        indices = indices.tolist()

        # Add the selected atom itself to the list of atoms to be deleted
        indices.append(index)

        # Delete atoms by their indices
        atoms = atoms[[atom.index for atom in atoms if atom.index not in indices]]

    return atoms

def get_pseudo_potential(xtal, pixel_size = 0.0725, sigma=0.2, axis_extent = None, type = 'Gaussian'):
    positions = xtal.get_positions()[:, :2]

    if axis_extent is not None:
        xmin, xmax, ymin, ymax = axis_extent
        x_coords = np.linspace(xmin, xmax, int((xmax - xmin) / pixel_size))
        y_coords = np.linspace(ymin, ymax, int((ymax - ymin) / pixel_size))
    else:
        xmin, xmax = np.min(positions[:, 0]), np.max(positions[:, 0])
        ymin, ymax = np.min(positions[:, 1]), np.max(positions[:, 1])
        fov = max(xmax - xmin, ymax - ymin)
        n_points = int(fov / pixel_size)
        x_coords = np.linspace(xmin, xmax, n_points)
        y_coords = np.linspace(ymin, ymax, n_points)

    # Map of atomic numbers - i.e. scattering intensity
    atomic_numbers = xtal.get_atomic_numbers() 
    potential_map = np.zeros((len(x_coords), len(y_coords)))
    for pos, atomic_number in zip(positions, atomic_numbers):
        x_idx = np.searchsorted(x_coords, pos[0])
        y_idx = np.searchsorted(y_coords, pos[1])
        potential_map[x_idx, y_idx] += atomic_number

    # Define differrent cross section shapes
    size = [10,10]
    epsilon = 1e-9

    # Gaussian kernel
    if type.lower() == 'gaussian':
        blurred_map = gaussian_filter(potential_map, sigma=1/sigma)

    # Coulombic Individual Atomic Model (IAM): 1/r
    if type.lower() == 'coulombic':
        x = np.arange(-size//2 + 1, size//2 + 1)
        y = np.arange(-size//2 + 1, size//2 + 1)
        xx, yy = np.meshgrid(x, y, sparse=True)
        r = np.sqrt(xx**2 + yy**2) + epsilon  # Add epsilon to avoid division by zero
        kernel = 1 / r

    # Electronic Shielding Effects

    normalized_map = blurred_map / np.max(blurred_map)

    return normalized_map.T


def get_masks(xtal, pixel_size=0.1, radius=3, axis_extent=None, mode='one_hot'):
    positions = xtal.get_positions()[:, :2]
    atomic_numbers = xtal.get_atomic_numbers()
    _, inverse_indices = np.unique(atomic_numbers, return_inverse=True)
    atom_ids = inverse_indices + 1  # the background pixels will be labeled as 0
    unique_atom_ids = np.unique(atom_ids)

    # Determine image size
    if axis_extent is not None:
        xmin, xmax, ymin, ymax = axis_extent
    else:
        xmin, xmax = np.min(positions[:, 0]), np.max(positions[:, 0])
        ymin, ymax = np.min(positions[:, 1]), np.max(positions[:, 1])
    img_height = int((ymax - ymin) / pixel_size)
    img_width = int((xmax - xmin) / pixel_size)

    master_mask = np.zeros((len(unique_atom_ids), img_height, img_width), dtype=np.uint8)
    
    def create_mask_for_atom(atom_id):
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        atom_mask = (atom_ids == atom_id)
        atom_positions = positions[atom_mask]

        # Make mask 1 in radius around each atom
        for x, y in atom_positions:
            x_pixel = int((x - xmin) / pixel_size)
            y_pixel = int((y - ymin) / pixel_size)
            rr, cc = disk((y_pixel, x_pixel), radius, shape=mask.shape)
            mask[rr, cc] = 1
        master_mask[atom_id - 1, mask == 1] = 1

    # Parallelize the mask creation
    tasks = [dask.delayed(create_mask_for_atom)(atom_id) for atom_id in unique_atom_ids]
    dask.compute(*tasks)

    if mode.lower() == 'one_hot':
        num_masks = unique_atom_ids.size + 1  # include background
        background_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        background_mask[(np.sum(master_mask, axis=0) == 0)] = 1
        masks = np.stack([background_mask] + [master_mask[i] for i in range(len(unique_atom_ids))], axis=0)
        return masks

    elif mode.lower() == 'binary':
        sum_masks = np.sum(master_mask, axis=0)
        final_mask = np.where(sum_masks > 0, 1, 0)
        return final_mask

    elif mode.lower() == 'integer':
        final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for i, mask in enumerate(master_mask):
            final_mask[mask == 1] = i + 1
        return final_mask

    else:
        raise ValueError("Invalid mode. Choose from 'one_hot', 'binary', or 'integer'")


def get_point_spread_function(airy_disk_radius = 1, size = 32):
    """
    Generate a Point Spread Function (PSF) for a Transmission Electron Microscope (TEM)
    using an Airy disk model.

    Parameters:
    - airy_disk_radius: Radius of the first Airy disk ring. This is a proxy for focus and aperture size.
    - size: Determines the size of the generated PSF array (total array size will be 2*size + 1).

    Returns:
    - psf: 2D NumPy array representing the PSF.
    """
    
    # Create grid
    x = np.arange(-size, size + 1)
    y = np.arange(-size, size + 1)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    # Calculate the Airy pattern (PSF)
    with np.errstate(divide='ignore', invalid='ignore'):
        psf = (2 * sp.j1(airy_disk_radius * rr) / (airy_disk_radius * rr))**2
        psf[rr == 0] = 1  # Handling the division by zero at the center

    # Normalize the PSF
    psf /= np.sum(psf)

    return psf


def convolve_kernel(potential, psf):
    """
    Simulate a TEM image by convolving the potential with the PSF.
    The potential and the PSF are first padded to compatible sizes.

    Parameters:
    - potential: 2D NumPy array representing the potential.
    - psf: 2D NumPy array representing the PSF.

    Returns:
    - image: 2D NumPy array representing the simulated TEM image.
    """

    # Sizes of the potential and the PSF
    potential_size = np.array(potential.shape)
    psf_size = np.array(psf.shape)

    # Size for padding - the sum of sizes minus 1 (to account for overlap)
    padded_size = potential_size + psf_size - 1

    # Pad the potential and the PSF to the padded size
    padded_potential = np.pad(potential, [(0, padded_size[0] - potential_size[0]), (0, padded_size[1] - potential_size[1])], mode='constant')
    padded_psf = np.pad(psf, [(0, padded_size[0] - psf_size[0]), (0, padded_size[1] - psf_size[1])], mode='constant')

    # Convolve the padded potential with the padded PSF using Fourier transform
    image = np.fft.ifft2(np.fft.fft2(padded_potential) * np.fft.fft2(padded_psf))

    # Normalize the image
    image = np.abs(image)
    image /= np.max(image)

    # Determine the cropping indices to extract the central part of the convolved image
    start_x = (padded_size[0] - potential_size[0]) // 2
    end_x = start_x + potential_size[0]
    start_y = (padded_size[1] - potential_size[1]) // 2
    end_y = start_y + potential_size[1]

    # Crop the image back to the original potential size
    image = image[start_x:end_x, start_y:end_y]

    return image


def add_poisson_noise(image, shot_noise=0.8):
    """
    Add Poisson noise to an image based on a sampling parameter.

    Parameters:
    - image: A 2D NumPy array representing the image.
    - shot_noise: A float representing the sampling parameter. A higher value results in more noise. 0 - 1

    Returns:
    - noisy_image: Image with Poisson noise added.
    """

    # Normalize the image
    image_normalized = image / np.max(image)

    # Adjust the scale of the noise
    noise = np.power(10, 10 * (1-shot_noise))

    # Generate Poisson noise
    noisy_image = np.random.poisson(image_normalized * noise)

    # Rescale to original range
    noisy_image = (noisy_image / np.max(noisy_image)) * np.max(image)

    return noisy_image


def grid_crop(image_master, crop_size=512, crop_glide=128):
    '''
    Slices an image into smaller, overlapping square crops.

    This function takes a larger image and divides it into smaller, overlapping square segments. 
    It's useful for processing large images in smaller batches, especially in machine learning applications 
    where input size is fixed.

    Parameters:
    - image_master: A NumPy array representing the image to be cropped. 
                    It should be a 2D array if the image is grayscale, or a 3D array for RGB images.
    - crop_size (int, optional): The size of each square crop. Default is 256 pixels.
    - crop_glide (int, optional): The stride or glide size for cropping. 
                                 Determines the overlap between consecutive crops. Default is 128 pixels.

    Returns:
    - cropped_ims: A NumPy array containing the cropped images. 
                   The array is 3D, where the first dimension represents the index of the crop, 
                   and the next two dimensions represent the height and width of the crops.

    Note:
    - The function assumes the input image is square. Non-square images might lead to unexpected results.
    - The return array is of type 'float16' to reduce memory usage, which might affect the precision of pixel values.
    '''

    n_crops = int((len(image_master) - crop_size)/crop_glide + 1)
    cropped_ims = np.zeros((n_crops,n_crops,crop_size,crop_size))

    for x in np.arange(n_crops):
        for y in np.arange(n_crops):
            xx,yy = int(x*crop_glide), int(y*crop_glide)
            cropped_ims[int(x),int(y)] = image_master[xx:xx+crop_size,yy:yy+crop_size]
    cropped_ims = cropped_ims.reshape((-1,crop_size,crop_size)).astype('float16')

    return cropped_ims


def resize_image(array, n, order = 3):
    """
    Resize a numpy array to n x n using interpolation.

    Parameters:
    array (numpy.ndarray): The input array.
    n (int): The size of the new square array.

    Returns:
    numpy.ndarray: The resized square array.
    """
    # Get the current shape of the array
    height, width = array.shape[-2:]

    # Calculate zoom factors
    zoom_factor = n / max(height, width)
    array = array.astype(np.float32)

    if len(array.shape) == 2:
        return zoom(array, [zoom_factor, zoom_factor], order = order)
    elif len(array.shape) == 3:
        return zoom(array, [1,zoom_factor, zoom_factor], order = order)


def shotgun_crop(image, crop_size=512, magnification_var = None, n_crops=10, seed=42, return_binary = False, roi = 'middle'):
    """
    Randomly crops a specified number of sub-images from a given image with variable magnification, supporting images with any number of channels.

    Parameters:
    image (numpy.ndarray): The input image as a NumPy array.
    crop_size (int, optional): The default size for each square crop. Defaults to 512.
    magnification_var (float, optional): The range of magnification variability as a fraction of the crop size. 
        If specified, each crop will be randomly sized within [crop_size * (1 - magnification_var), crop_size * (1 + magnification_var)]. Defaults to None.
    n_crops (int, optional): The number of crops to generate. Defaults to 10.
    seed (int, optional): Seed for the random number generator for reproducibility. Uses random package.

    Returns:
    numpy.ndarray: An array containing the cropped (and potentially resized) images as NumPy arrays.

    Important:
    If using this funciton on an image and mask together, make sure to use the same seed for both.
    """

    if return_binary == True:
        order = 0
    else:
        order = 3

    # Set seed for reproducibility
    # Seed should be a very large integer for good results
    crop_rng = np.random.default_rng(seed)

    # Get crop sizes for changing magnification later
    if magnification_var is not None:
        crop_sizes = crop_rng.integers(crop_size * ( 1 - magnification_var), crop_size * (1 + magnification_var), n_crops)
        crop_sizes = crop_sizes.astype(int)
    else:
        crop_sizes = np.full(n_crops, crop_size)

    # Randomly crop images (position and size)
    h, w = image.shape[-2:]
    crops = []
    for size in crop_sizes:
        if roi == 'middle':
            edge_cutoff = 256
            top = crop_rng.integers(edge_cutoff, h - size - edge_cutoff)
            left = crop_rng.integers(edge_cutoff, w - size - edge_cutoff)
        else:
            top = crop_rng.integers(0, h - size)
            left = crop_rng.integers(0, w - size)
        if len(image.shape) > 2:
            crop = image[:, top:top+size, left:left+size]
            crop = resize_image(crop, crop_size, order)
        else:
            crop = image[top:top+size, left:left+size]
            crop = resize_image(crop, crop_size, order)
        crops.append(crop)

    crops = np.array(crops)
    batch_crops = np.stack(crops, axis=0)
        
    return batch_crops
