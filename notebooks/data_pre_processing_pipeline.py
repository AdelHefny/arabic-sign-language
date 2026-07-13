def gather_tasks(directories, dest_dir):
    """Crawls directories and builds a list of inputs and outputs for the workers."""
    tasks = []

    for d in directories:
        if not os.path.exists(d):
            continue

        classes = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]

        for class_name in classes:
            class_source_path = os.path.join(d, class_name)
            class_dest_path = os.path.join(dest_dir, class_name)

            os.makedirs(class_dest_path, exist_ok=True)

            videos = glob.glob(os.path.join(class_source_path, '*.mp4'))

            for video_path in videos:
                video_filename = os.path.basename(video_path)
                output_path = os.path.join(class_dest_path, video_filename.replace('.mp4', '.npy'))

                # Only add to task queue if not already processed
                if not os.path.exists(output_path):
                    tasks.append((video_path, output_path))

    return tasks
import os
import glob
base_path = '/content/drive/MyDrive/arabic sign language'

COLAB_OUT_DIR_train = '/content/npy_dataset/train'
COLAB_OUT_DIR_test = '/content/npy_dataset/test'

train_dirs = [
    os.path.join(base_path, 'extracted_02_train'),
]

test_dirs = [
    os.path.join(base_path, 'extracted_02_test'),
]

print("--- Gathering Training Tasks ---")
train_tasks = gather_tasks(train_dirs, COLAB_OUT_DIR_train)
train_tasks = train_tasks[:len(train_tasks)//2]
print(len(train_tasks))
print("\n--- Gathering Testing Tasks ---")
test_tasks = gather_tasks(test_dirs, COLAB_OUT_DIR_test)
test_tasks = test_tasks[:len(test_tasks)//2]
print(len(test_tasks))
!wget -q -O /content/holistic_landmarker.task https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import concurrent.futures
from tqdm.notebook import tqdm
import os

# ==========================================
# 1. EXTRACTION & TEMPORAL SMOOTHING
# ==========================================

def extract_frame_landmarks(result):
    """
    Extracts Pose, Left Hand, and Right Hand from the Tasks API HolisticLandmarkerResult.
    If a body part is missing, it fills the array with NaNs (Not a Number)
    instead of zeros for temporal interpolation.
    """
    # 1. Pose (33 points)
    if result.pose_landmarks:
        # The Tasks API returns a flat List[NormalizedLandmark] for holistic pose
        pose = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks])
    else:
        pose = np.full((33, 3), np.nan)

    # 2. Left Hand (21 points)
    if result.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks])
    else:
        lh = np.full((21, 3), np.nan)

    # 3. Right Hand (21 points)
    if result.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks])
    else:
        rh = np.full((21, 3), np.nan)

    return np.concatenate([pose, lh, rh]) # Shape: (75, 3)

def interpolate_and_normalize(frames_array):
    """
    Fixes the 5 major flaws:
    - Flaw 3: Interpolates missing frames instead of using hard zeros.
    - Flaw 1: Sets a global origin (Mid-Shoulder) instead of independent wrists.
    - Flaw 2: Sets a global scale (Shoulder Width) instead of independent hand sizes.
    """
    shape = frames_array.shape

    # --- TEMPORAL INTERPOLATION ---
    df = pd.DataFrame(frames_array.reshape(shape[0], -1))
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    frames_array = df.to_numpy().reshape(shape)

    if np.isnan(frames_array).all():
        return None

    # --- GLOBAL ANCHOR ---
    left_shoulder = frames_array[:, 11, :]
    right_shoulder = frames_array[:, 12, :]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    centered_frames = frames_array - mid_shoulder[:, np.newaxis, :]

    # --- GLOBAL SCALING ---
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    shoulder_width[shoulder_width == 0] = 1e-6
    normalized_frames = centered_frames / shoulder_width[:, np.newaxis, np.newaxis]

    # Flatten the final dimensions to (num_frames, 225)
    return normalized_frames.reshape(shape[0], -1)


# ==========================================
# 2. BATCH WORKER FUNCTION
# ==========================================

def process_single_video(video_path, output_path, model_path='holistic_landmarker.task'):
    """Processes a single video utilizing the new Tasks API."""
    try:
        # 1. Setup Tasks API Options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5
        )

        frames_data = []
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0

        with vision.HolisticLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Keep frame skipping logic only if your native video is >= 30 FPS
                if frame_index % 2 != 0:
                    frame_index += 1
                    continue

                frame = cv2.resize(frame, (640, 480))
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                # Convert to Tasks API Image format
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                # Calculate strictly increasing timestamp in milliseconds
                timestamp_ms = int(frame_index * (1000 / fps))

                # Process the frame via Tasks API
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                frames_data.append(extract_frame_landmarks(result))

                frame_index += 1

        cap.release()

        if len(frames_data) == 0:
            return False, f"Empty video: {video_path}"

        frames_array = np.array(frames_data)

        # Apply smoothing and spatial normalization
        processed_sequence = interpolate_and_normalize(frames_array)

        if processed_sequence is None:
            return False, f"No human detected in: {video_path}"

        # Save output
        np.save(output_path, processed_sequence.astype(np.float16))
        return True, video_path

    except Exception as e:
        return False, f"Error on {os.path.basename(video_path)}: {e}"


# ==========================================
# 3. MULTI-CORE EXECUTION ENGINE
# ==========================================

def run_parallel_pipeline(tasks):
    """Executes the pipeline across all Azure ML CPU cores."""
    if not tasks:
        print("No tasks provided.")
        return

    max_workers = os.cpu_count()
    print(f"Igniting ProcessPoolExecutor with {max_workers} cores...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Note: Ensure you provide the correct path to holistic_landmarker.task in tasks definition
        futures = [executor.submit(process_single_video, t[0], t[1]) for t in tasks]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Extracting Tasks API Landmarks"):
            success, message = future.result()
            if not success:
                print(message)
print(f"Found {len(train_tasks)} training tasks.")
run_parallel_pipeline(train_tasks)

print(f"\nFound {len(test_tasks)} testing tasks.")
run_parallel_pipeline(test_tasks)


import os
import json
from google.colab import userdata

# --- CHANGE THIS FOR EACH NOTEBOOK (1, 2, or 3) ---
PART_NUMBER = 1
# -------------------------------------------------

# 1. Setup API Credentials (assuming you added them to Colab Secrets)
!pip install kaggle -q
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

DATASET_PATH = "/content/npy_dataset"

# 2. Generate Unique Metadata for this Part
metadata = {
  "title": f"Arabic Sign Language Landmarks NPY - Part {PART_NUMBER}",
  # ID must be lowercase, alphanumeric, and dashes only
  "id": f"{os.environ['KAGGLE_USERNAME']}/arsl-landmarks-npy-part{PART_NUMBER}",
  "licenses": [{"name": "CC0-1.0"}]
}

with open(os.path.join(DATASET_PATH, "dataset-metadata.json"), "w") as f:
    json.dump(metadata, f)

# 3. Authenticate and Upload
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

print(f"Compressing and uploading Part {PART_NUMBER} to Kaggle...")

# Create a brand new dataset for this specific chunk
api.dataset_create_new(
    folder=DATASET_PATH,
    dir_mode='zip',
    convert_to_csv=False,
    public=False
)

print(f"Part {PART_NUMBER} upload successful!")

# Using /content as it is writable in Colab
BASE_PATH = "/content/kaggle_datasets"
!mkdir -p {BASE_PATH}/part1
!mkdir -p {BASE_PATH}/part2

# Download and Unzip if credentials are set
if 'KAGGLE_USERNAME' in os.environ:
    !kaggle datasets download -d adelhefny/arsl-landmarks-npy-part1
    !unzip -q arsl-landmarks-npy-part1.zip -d {BASE_PATH}/part1
    !kaggle datasets download -d adelhefny/arsl-landmarks-npy-part2
    !unzip -q arsl-landmarks-npy-part2.zip -d {BASE_PATH}/part2
    !rm *.zip
    print("Datasets downloaded and extracted successfully.")
else:
    print("Skipping download: Kaggle credentials not found.")
!kaggle datasets download -d adelhefny/arsl-landmarks-npy-part3
!unzip -q arsl-landmarks-npy-part3.zip -d {BASE_PATH}/part3
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Using /content as it is writable in Colab
BASE_PATH = "/content/kaggle_datasets"
base_dirs = [
    os.path.join(BASE_PATH, "part2"),
    os.path.join(BASE_PATH, "part3"),
    os.path.join(BASE_PATH, "part1"),
]

def get_paths_and_labels(base_dirs, data_type):
    paths, labels = [], []
    # Check if the first base_dir and its 'train' subdirectory exist
    # before trying to list its contents.
    train_path_in_first_base_dir = os.path.join(base_dirs[0], "train")
    if not os.path.exists(train_path_in_first_base_dir):
        # Handle the case where the expected directory doesn't exist
        # This could happen if a part of the dataset failed to download/unzip
        print(f"Warning: {train_path_in_first_base_dir} not found. Skipping dataset processing.")
        return [], [], 0 # Return empty lists and 0 classes

    unique_labels = sorted(os.listdir(train_path_in_first_base_dir))
    label_to_index = {name: i for i, name in enumerate(unique_labels)}

    for base_dir in base_dirs:
        data_path = os.path.join(base_dir, data_type)
        if not os.path.exists(data_path): continue

        for label in unique_labels:
            label_dir = os.path.join(data_path, label)
            if os.path.exists(label_dir):
                for file in os.listdir(label_dir):
                    paths.append(os.path.join(label_dir, file))
                    labels.append(label_to_index[label])
    return paths, labels, len(unique_labels)

train_val_paths, train_val_labels, num_classes = get_paths_and_labels(base_dirs, "train")

# Only proceed with splitting if data was found
if train_val_paths:
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_val_labels
    )
else:
    print("No training data found. Skipping train/val split.")
    train_paths, val_paths, train_labels, val_labels = [], [], [], []

test_paths, test_labels, _ = get_paths_and_labels(base_dirs, "test")
import tensorflow as tf
import numpy as np
import tqdm
import os

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord(paths, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for path, label in tqdm.tqdm(zip(paths, labels), total=len(paths)):
            try:
                data = np.load(path).astype(np.float32)
                data = np.nan_to_num(data, nan=0.0)

                # Changing mask value to 1.0
                mask = np.all(data == 0, axis=-1)
                data[mask] = 1.0

                flattened_data = data.flatten().tolist()
                num_frames = data.shape[0]

                feature = {
                    'data': _float_feature(flattened_data),
                    'label': _int64_feature(label),
                    'num_frames': _int64_feature(num_frames)
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            except Exception as e:
                print(f'Error processing {path}: {e}')

if not os.path.exists('tfrecords'):
    os.makedirs('tfrecords')

# Set current expected dimension for 75 landmarks (x,y,z)
CURRENT_FEATURES = 225

print(f'Converting datasets to TFRecords (Features: {CURRENT_FEATURES}) with mask_value=1.0...')
create_tfrecord(train_paths, train_labels, 'tfrecords/train.tfrecord')
create_tfrecord(val_paths, val_labels, 'tfrecords/val.tfrecord')
create_tfrecord(test_paths, test_labels, 'tfrecords/test.tfrecord')
print('TFRecord creation complete.')
def parse_tfrecord_fn(example):
    feature_description = {
        'data': tf.io.VarLenFeature(tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'num_frames': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)

    data = tf.sparse.to_dense(example['data'])
    num_frames = tf.cast(example['num_frames'], tf.int32)

    # Strictly use 225 features for the new landmark set
    data = tf.reshape(data, [num_frames, 225])

    # Use 1.0 as the sentinel value for padding/missing data
    data = tf.where(tf.math.is_nan(data), tf.ones_like(data), data)
    is_padding = tf.reduce_all(tf.equal(data, 0.0), axis=-1, keepdims=True)
    data = tf.where(tf.repeat(is_padding, 225, axis=-1), 1.0 * tf.ones_like(data), data)

    label = tf.one_hot(tf.cast(example['label'], tf.int32), depth=num_classes)
    return data, label

def load_tfrecord_dataset(file_path, batch_size=32, shuffle=True):
    ds = tf.data.TFRecordDataset(file_path)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # Updated padded_batch shape to 225
    ds = ds.padded_batch(batch_size, padded_shapes=([None, 225], [num_classes]), padding_values=(1.0, 0.0))
    return ds.prefetch(tf.data.AUTOTUNE)