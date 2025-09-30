import logging
import os
from datetime import datetime

# 1. Define JUST the directory path
logs_dir = os.path.join(os.getcwd(), "logs")

# 2. Create that directory
os.makedirs(logs_dir, exist_ok=True)

# 3. Define the filename
log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# 4. Now, create the full file path by joining the DIRECTORY and the FILENAME
log_file_path = os.path.join(logs_dir, log_file_name)

# 5. This path is now correct and can be used in the config
logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)