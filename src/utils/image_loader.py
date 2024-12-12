import sys
import os
from pathlib import Path
from loguru import logger

class ImageLoader:
    """
    Image Loader capable of finding and getting all images paths in a given folder recursively
    in order to load them into a model
    """
    def __init__(self, folder, verbose=False):

        # Attr initialization
        self.folder = Path(folder)

        # Setup logging
        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Checking folder
        try:
            logger.info("Checking provided path.")
            if not self.folder.is_dir():
                raise NotADirectoryError(f"Provided path {self.folder.absolute()} is not a directory")
            logger.info(f"Provided path {self.folder.absolute()} is a valid directory.")
        except NotADirectoryError as e:
            logger.error(f"Error: {e}")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f"Couldnt find provided path: {self.folder}")
            sys.exit(1)
        except PermissionError:
            logger.error(f"No permission to access provided path: {self.folder}")
            sys.exit(1)



    def find_images(self):
        """Find all image files in path recursively. Return their paths"""
        # return (
        #     list(self.folder.rglob("*.jpg"))
        #     + list(self.folder.rglob("*.JPG"))
        #     + list(self.folder.rglob("*.jpeg"))
        #     + list(self.folder.rglob("*.png"))
        #     + list(self.folder.rglob("*.bmp"))
        # )
        
        # We need to avoid using JPG and jpg (upper and lower)
        # bacause of Unix systems (Case sensitive) and NTFS Systems (Non Case Sensitive).
        # This will avoid duplicates 
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        # Find all image files recursively and filter by extension (lowercase only)
        image_paths = [img_path for img_path in self.folder.rglob('*') if img_path.suffix.lower() in image_extensions]
        # Convert to lowercase and remove duplicates (especially relevant for Windows)
        unique_image_paths = {img_path.resolve().as_posix().lower(): img_path for img_path in image_paths}
        
        return list(unique_image_paths.values())

