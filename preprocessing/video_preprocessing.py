import os 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import dlib
import time 
import warnings 
import math
from scipy.spatial import distance
import joblib
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog 
from skimage.color import rgb2gray
warnings.filterwarnings("ignore")

from data_setup import EMOTION_MAP, PROJECT_ROOT, DATA_PATH, PROCESSED_PATH

print("initializing video preprocessing...")

# initialize dlib's face detector 
face_detector = dlib.get_frontal_face_detector()

SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"

landmark_predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))

print("initializing video preprocessing done.")

def extract_frames(video_path, sample_rate = 5, max_frames = None):
    # Extract frames from a video file at a specific sample rate
    # video_path (str): Path to the video file
    # sample_rate (int): Extract every Nth frame
    # max_frames (int, optional): Maximum number of frames to extract

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return [], 0, {}

    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    frame_interval = max(1, int(sample_rate))

    video_metadata = {
        'path': video_path,
        'filename': os.path.basename(video_path),
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': duration,
        'sample_rate': sample_rate
    }

    #extract metadata from filename of RAVDESS

    filename = video_metadata['filename']
    parts = filename.split('-')
    emotion_labels = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    if len(parts) >= 7:
        try:
            video_metadata['modality'] = parts[1]
            video_metadata['vocal_channel'] = parts[2]
            video_metadata['emotion'] = EMOTION_MAP.get(parts[3], 'unknown')
            video_metadata['emotion_code'] = parts[3]
            video_metadata['intensity'] = 'normal' if parts[4] == '01' else 'strong'
            video_metadata['statement'] = parts[5]
            video_metadata['repetition'] = parts[6]
            actor_id = parts[7].split('.')[0]
            video_metadata['actor_id'] = actor_id
            video_metadata['gender'] = 'female' if int(actor_id) % 2 == 0 else 'male'
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            # Add default values for required keys if parsing fails
            video_metadata.setdefault('modality', 'unknown')
            video_metadata.setdefault('vocal_channel', 'unknown')
            video_metadata.setdefault('emotion', 'unknown')
            video_metadata.setdefault('emotion_code', 'unknown')
            video_metadata.setdefault('intensity', 'unknown')
            video_metadata.setdefault('statement', 'unknown')
            video_metadata.setdefault('repetition', 'unknown')
            video_metadata.setdefault('actor_id', 'unknown')
            video_metadata.setdefault('gender', 'unknown')


    frames = []
    frame_indices = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_indices.append(frame_index)

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_index += 1

    cap.release()

    video_metadata['extracted_frames'] = len(frames)
    video_metadata['frame_indices'] = frame_indices

    print(f"Extracted {len(frames)} frames from {video_path} at a sample rate of {sample_rate}.")
    return frames, frame_count, video_metadata


def detect_faces_landmarks(frame):
    # Detect faces and landmarks in a single frame using Dlib

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray, 0)

    if len(faces) == 0:
        return  False, None, None, 0.0
    
    face = faces[0]

    detection_confidence = 1.0

    # Get the landmarks/parts for the face in box d.
    face_bbox = (face.left(), face.top(), face.right(), face.bottom())

    shape = landmark_predictor(gray, face)

    # # Convert landmarks to a list of (x, y, z) tuples

    h, w = gray.shape
    landmarks = []

    for i in range(68):
        x = shape.part(i).x /w # normaize to [0, 1]
        y = shape.part(i).y /h
        landmarks.append((x, y, 0.0)) # z is set to 0.0

    return True, landmarks, face_bbox, detection_confidence

def align_face(frame, landmarks, bbox, target_size = (224, 224)):
    # Align the face in the frame using landmarks and bounding box
    
    x_min, y_min, x_max, y_max = bbox

    # add margin around face
    h, w = frame.shape[:2]
    margin_x = int((x_max - x_min) * 0.2)
    margin_y = int((y_max - y_min) * 0.2)

    y_min = max(0, y_min - margin_y)
    y_max = min(h, y_max + margin_y)
    x_min = max(0, x_min - margin_x)
    x_max = min(w, x_max + margin_x)

    face_img = frame[y_min:y_max, x_min:x_max]

    if face_img.size > 0:
        face_img = cv2.resize(face_img, target_size)
    else :
        face_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    #  L* representing lightness, a* representing greenness to redness, and b* representing blueness to yellowness
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    #convert back to RGB
    enhanced_face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced_face    

def extract_geometric_features(landmarks):
    #extract geometric features from landmarks

    features = {}

    #convert landmark to np array
    landmarks_array = np.array(landmarks)

    left_eye = list(range(42, 48))
    right_eye = list(range(36, 42))
    left_eyebrow = list(range(22, 27))
    right_eyebrow = list(range(17, 22))
    mouth_outline = list(range(48, 60))
    inner_mouth = list(range(60, 68))
    nose_tip = 30
    nose_bottom = 33
    nose_right = 35
    nose_left = 31
    jaw_line = list(range(0, 17))

    #calculate featuree

    # 1. eye aspect ratio (EAR) - meansures eye openness
    def eye_aspect_ratio(eye_landmarks):
        #vertical dist
        vert1 = distance.euclidean(landmarks[eye_landmarks[1]], landmarks[eye_landmarks[5]])
        vert2 = distance.euclidean(landmarks[eye_landmarks[2]], landmarks[eye_landmarks[4]])

        #horizontal dist
        horiz = distance.euclidean(landmarks[eye_landmarks[0]], landmarks[eye_landmarks[3]])

        # calculate EAR
        ear = (vert1 + vert2) / (2.0 * horiz) if horiz > 0 else 0
        return ear
    
    features['left_eye_ear'] = eye_aspect_ratio(left_eye)
    features['right_eye_ear'] = eye_aspect_ratio(right_eye)
    features['avg_eye_ear'] = (features['left_eye_ear'] + features['right_eye_ear']) / 2.0

    # 2. mouth aspect ratio (MAR) - measures mouth openness
    mouth_top = landmarks_array[mouth_outline[3]]  # Upper lip
    mouth_bottom = landmarks_array[mouth_outline[9]]  # Lower lip
    mouth_left = landmarks_array[mouth_outline[0]]  # Left corner
    mouth_right = landmarks_array[mouth_outline[6]]  # Right corner

    mouth_width = distance.euclidean(mouth_left, mouth_right)
    mouth_height = distance.euclidean(mouth_top, mouth_bottom)

    features['mouth_aspect_ratio'] = mouth_height / mouth_width if mouth_width > 0 else 0

    # 3. eyebrow position relative to eyes
    left_eye_center = np.mean(landmarks_array[left_eye], axis=0)
    right_eye_center = np.mean(landmarks_array[right_eye], axis=0)
    
    left_eyebrow_center = np.mean(landmarks_array[left_eyebrow], axis=0)
    right_eyebrow_center = np.mean(landmarks_array[right_eyebrow], axis=0)
    
    features['left_eyebrow_eye_dist'] = left_eyebrow_center[1] - left_eye_center[1]
    features['right_eyebrow_eye_dist'] = right_eyebrow_center[1] - right_eye_center[1]
    
    # 4. Mouth width to face width ratio
    face_width = distance.euclidean(landmarks_array[jaw_line[0]], landmarks_array[jaw_line[16]])
    features['mouth_face_width_ratio'] = mouth_width / face_width if face_width > 0 else 0
    
    # 5. Smile ratio (measure of mouth curvature)
    mouth_corner_left = landmarks_array[mouth_outline[0]]
    mouth_corner_right = landmarks_array[mouth_outline[6]]
    mouth_center_top = landmarks_array[mouth_outline[3]]
    
    # Calculate smile curve (higher values for upward curves)
    mouth_curve = (mouth_corner_left[1] + mouth_corner_right[1]) / 2 - mouth_center_top[1]
    features['smile_ratio'] = mouth_curve / mouth_width if mouth_width > 0 else 0
    
    # 6. Nose wrinkle (distance between eyebrows and nose)
    nose_bridge = landmarks_array[nose_tip]
    between_eyebrows = (landmarks_array[left_eyebrow[2]] + landmarks_array[right_eyebrow[2]]) / 2
    features['nose_wrinkle'] = distance.euclidean(nose_bridge, between_eyebrows)
    
    # 7. Eye openness ratio
    features['eye_openness_ratio'] = features['avg_eye_ear'] / features['mouth_aspect_ratio'] if features['mouth_aspect_ratio'] > 0 else 0
    
    # 8. Asymmetry measures
    features['eye_asymmetry'] = abs(features['left_eye_ear'] - features['right_eye_ear'])
    features['eyebrow_asymmetry'] = abs(features['left_eyebrow_eye_dist'] - features['right_eyebrow_eye_dist'])

    # 9. Jawline tension
    jaw_left = landmarks_array[jaw_line[0]]
    jaw_right = landmarks_array[jaw_line[16]]
    jaw_bottom = landmarks_array[jaw_line[8]]
    
    # Calculate jaw angle
    jaw_leftside = distance.euclidean(jaw_left, jaw_bottom)
    jaw_rightside = distance.euclidean(jaw_right, jaw_bottom)
    features['jaw_asymmetry'] = abs(jaw_leftside - jaw_rightside) / (jaw_leftside + jaw_rightside) if (jaw_leftside + jaw_rightside) > 0 else 0
    
    # 10. Inner mouth openness
    inner_mouth_top = landmarks_array[inner_mouth[3]]
    inner_mouth_bottom = landmarks_array[inner_mouth[1]]
    inner_mouth_height = distance.euclidean(inner_mouth_top, inner_mouth_bottom)
    features['inner_mouth_openness'] = inner_mouth_height / mouth_height if mouth_height > 0 else 0
    
    return features

def extract_appearance_features(aligned_face, hog_orientation = 9, hog_pixel_per_cell = (8, 8), hog_cells_per_block = (2, 2)):
    # Extract appearance features from the aligned face
    #hog_orientation (int): Number of orientation bins for HOG
    #hog_pixel_per_cell (tuple): Size of each cell in pixels for HOG
    #hog_cells_per_block (tuple): Number of cells per block for HOG

    features = {}

    gray_face = rgb2gray(aligned_face)

    # Compute HOG features
    hog_features, hog_image = hog(gray_face, orientations = hog_orientation, pixels_per_cell = hog_pixel_per_cell, cells_per_block = hog_cells_per_block, visualize = True, block_norm = "L2-Hys")
    features['hog_mean'] = np.mean(hog_features)
    features['hog_std'] = np.std(hog_features)
    features['hog_min'] = np.min(hog_features)
    features['hog_max'] = np.max(hog_features)
    
    # We also compute some basic image statistics
    features['face_brightness'] = np.mean(gray_face)
    features['face_contrast'] = np.std(gray_face)
    
    # Edge intensity
    sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    features['edge_intensity_mean'] = np.mean(edge_magnitude)
    features['edge_intensity_std'] = np.std(edge_magnitude)
    
    return features, hog_image

def draw_landmarks(frame, landmarks, bbox):
    # draw facial landmarks and bouding box on the frame

    vis_frame = frame.copy()

    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    h, w = frame.shape[:2]
    for i, landmark in enumerate(landmarks):
        x, y = int(landmark[0] * w), int(landmark[1] * h)
        cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), -1)

    # Connect landmarks to show facial features
    # Jaw line
    for i in range(16):
        pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        pt2 = (int(landmarks[i + 1][0] * w), int(landmarks[i + 1][1] * h))
        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    
    # Right eyebrow
    for i in range(17, 21):
        pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        pt2 = (int(landmarks[i + 1][0] * w), int(landmarks[i + 1][1] * h))
        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    
    # Left eyebrow
    for i in range(22, 26):
        pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        pt2 = (int(landmarks[i + 1][0] * w), int(landmarks[i + 1][1] * h))
        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    
    # Nose bridge
    for i in range(27, 30):
        pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        pt2 = (int(landmarks[i + 1][0] * w), int(landmarks[i + 1][1] * h))
        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    
    # Nose bottom
    for i in range(31, 35):
        pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        pt2 = (int(landmarks[i + 1][0] * w), int(landmarks[i + 1][1] * h))
        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    
    # Connect nose bottom to nose bridge
    pt1 = (int(landmarks[30][0] * w), int(landmarks[30][1] * h))
    pt2 = (int(landmarks[35][0] * w), int(landmarks[35][1] * h))
    cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)

    # Right eye
    for i in range(36, 41):
        pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        pt2 = (int(landmarks[i + 1][0] * w), int(landmarks[i + 1][1] * h))
        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    # Close right eye
    pt1 = (int(landmarks[36][0] * w), int(landmarks[36][1] * h))
    pt2 = (int(landmarks[41][0] * w), int(landmarks[41][1] * h))
    cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    
    # Left eye
    for i in range(42, 47):
        pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        pt2 = (int(landmarks[i + 1][0] * w), int(landmarks[i + 1][1] * h))
        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    # Close left eye
    pt1 = (int(landmarks[42][0] * w), int(landmarks[42][1] * h))
    pt2 = (int(landmarks[47][0] * w), int(landmarks[47][1] * h))
    cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    
    # Outer mouth
    for i in range(48, 59):
        pt1 = (int(landmarks[i][0] * w), int(landmarks[i][1] * h))
        pt2 = (int(landmarks[i + 1][0] * w), int(landmarks[i + 1][1] * h))
        cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    # Close outer mouth
    pt1 = (int(landmarks[48][0] * w), int(landmarks[48][1] * h))
    pt2 = (int(landmarks[59][0] * w), int(landmarks[59][1] * h))
    cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 1)
    
    return vis_frame
    
def process_video(video_path, output_dir, visualize = False, sample_rate = 5, max_frame = None):
    # Process a single video file extract rames, detect faces, extract features


    frames, frame_count, video_metadata = extract_frames(video_path, sample_rate, max_frame)

    if not frames: 
        print(f"No frames extracted from {video_path}.")
        return None, video_metadata
    

    all_features =[]
    face_detected_count = 0

    if visualize: 
        vis_dir = Path(output_dir) / "visualization" / os.path.basename(video_path).split('.')[0]
        os.makedirs(vis_dir, exist_ok=True)

    for i, frame in enumerate(tqdm(frames, desc = f"Processing {os.path.basename(video_path)}")):
        frame_features = {
            'frame_index' : video_metadata['frame_indices'][i],
            'timestamp' : video_metadata['frame_indices'][i] / video_metadata['fps']
        }

        landmarks_detected, landmarks, bbox, confidence = detect_faces_landmarks(frame)

        if landmarks_detected:
            face_detected_count += 1
            frame_features['face_detected'] = True
            frame_features['detection_confidence'] = confidence

            aligned_face = align_face(frame, landmarks, bbox)

            geometric_features = extract_geometric_features(landmarks)
            frame_features.update(geometric_features)

            appearance_features, hog_image = extract_appearance_features(aligned_face)
            frame_features.update(appearance_features)

            if visualize and i % 10 == 0:
                vis_frame = draw_landmarks(frame, landmarks, bbox)

                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

                plt.figure(figsize=(15, 10))
                
                # Original frame with landmarks
                plt.subplot(2, 2, 1)
                plt.imshow(vis_frame)
                plt.title('Face Detection & Landmarks')
                plt.axis('off')
                
                # Aligned face
                plt.subplot(2, 2, 2)
                plt.imshow(aligned_face)
                plt.title('Aligned Face')
                plt.axis('off')
                
                # HOG visualization
                plt.subplot(2, 2, 3)
                plt.imshow(hog_image, cmap='gray')
                plt.title('HOG Features')
                plt.axis('off')
                
                # Feature values
                plt.subplot(2, 2, 4)
                features_to_plot = {
                    'Mouth AR': geometric_features['mouth_aspect_ratio'],
                    'Eye AR': geometric_features['avg_eye_ear'],
                    'Smile Ratio': geometric_features['smile_ratio'],
                    'Eyebrow Height': geometric_features['left_eyebrow_eye_dist']
                }
                plt.bar(features_to_plot.keys(), features_to_plot.values())
                plt.title('Key Geometric Features')
                plt.tight_layout()
                
                # Save the visualization
                plt.savefig(vis_dir / f"frame_{i:04d}.png")
                plt.close()
            

        else:
            frame_features['face_detected'] = False
            frame_features['detection_confidence'] = 0

        all_features.append(frame_features)
    
    video_metadata['face_detected_count'] = face_detected_count
    video_metadata['face_detection_rate'] = face_detected_count / len(frames) if frames else 0

    return all_features, video_metadata


def process_video_dataset(video_dir, output_dir, sample_rate = 5, max_videos = None, visualize_subset = True):
    # process all videos in directory

    os.makedirs(output_dir, exist_ok=True)

    video_extension = '.mp4'
    video_files = []

    video_files.extend(Path(video_dir).rglob(f"*{video_extension}"))

    video_files.sort()

    if max_videos is not None:
        video_files = video_files[:max_videos]

    print(f"Found {len(video_files)} videos in {video_dir}")

    all_video_features = []
    all_metadata = []

    for i, video_path in enumerate(video_files):
        vis_this_video = visualize_subset and i % 10 == 0

        print(f"\nProcessing video {i + 1}/{len(video_files)}: {video_path}")

        features_list, metadata = process_video(str(video_path), output_dir, visualize = vis_this_video, sample_rate = sample_rate)

        if not features_list:
            print(f"No features extracted from {video_path}.")
            continue

        video_id = i
        for features in features_list:
            features['video_id'] = video_id
            
            features['emotion'] = metadata['emotion']
            features['actor_id'] = metadata['actor_id']
            features['gender'] = metadata['gender']
            features['intensity'] = metadata['intensity']

            all_video_features.append(features)

        metadata['video_id'] = video_id
        all_metadata.append(metadata)

    # Convert to dataframe
    features_df = pd.DataFrame(all_video_features)
    metadata_df = pd.DataFrame(all_metadata)

    # Save to CSV (if not empty)
    if not features_df.empty:
        features_df.to_csv(Path(output_dir) / "video_features.csv", index = False)
        metadata_df.to_csv(Path(output_dir) / "video_metadata.csv", index = False)
        
        # Check if 'face_detected' column exists before filtering
        if 'face_detected' in features_df.columns:
            # Standardize the features
            numeric_features = features_df.select_dtypes(include=[np.number])
            numeric_features = numeric_features[features_df['face_detected'] == True]
            
            if len(numeric_features) > 0:
                cols_to_exclude = ['frame_index', 'timestamp', 'video_id', 'face_detected']
                feature_cols = [col for col in numeric_features.columns if col not in cols_to_exclude]

                features_for_scaling = numeric_features[feature_cols].fillna(0)

                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features_for_scaling)

                # Save the scaler for later use
                joblib.dump(scaler, os.path.join(output_dir, "video_feature_scaler.pkl"))

                scaled_df = pd.DataFrame(scaled_features, columns = feature_cols)

                for col in ['emotion', 'actor_id', 'gender', 'intensity']:
                    frame_indices = numeric_features.index
                    scaled_df[col] = features_df.loc[frame_indices, col].values
                
                scaled_df.to_csv(os.path.join(output_dir, "scaled_video_features.csv"), index = False)
        
        if len(features_df) > 0:
            feature_stats = features_df.describe()
            feature_stats.to_csv(os.path.join(output_dir, "video_feature_statistics.csv"))

            emotion_counts = features_df['emotion'].value_counts()

            plt.figure(figsize=(12, 6))
            emotion_counts.plot(kind='bar')
            plt.title('Frame Count by Emotion')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'emotion_distribution.png'))
            plt.close()
            
            # Face detection success rate by emotion
            plt.figure(figsize=(12, 6))
            face_detect_by_emotion = features_df.groupby('emotion')['face_detected'].mean()
            face_detect_by_emotion.plot(kind='bar')
            plt.title('Face Detection Success Rate by Emotion')
            plt.ylabel('Success Rate')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'face_detection_by_emotion.png'))
            plt.close()
    else:
        print("No features were extracted from any videos. Check your data or parameters.")
    
    print(f"\nVideo preprocessing complete! Processed {len(metadata_df)} videos.")
    return features_df, metadata_df

if __name__ == "__main__":
    from data_setup import DATA_PATH, PROCESSED_PATH
    import time


    VIDEO_SPEECH_PATH = DATA_PATH / "Video_Speech_Actors_01-24"
    VIDEO_SONG_PATH = DATA_PATH / "Video_Song_Actors_01-24"

    speech_output_dir = PROCESSED_PATH / "video_features" / "speech"
    song_output_dir = PROCESSED_PATH / "video_features" / "song"

    os.makedirs(speech_output_dir, exist_ok=True)
    os.makedirs(song_output_dir, exist_ok=True)

    print(f"Starting video preprocessing for SPEECH videos from {VIDEO_SPEECH_PATH}...")



    print(f"Starting video preprocessing for SPEECH videos from {VIDEO_SPEECH_PATH}...")
    start_time = time.time()
    speech_features, speech_metadata = process_video_dataset(
        VIDEO_SPEECH_PATH,
        speech_output_dir,
        sample_rate=5,  
        visualize_subset=True 
    )
    speech_time = time.time() - start_time
    print(f"Speech video preprocessing completed in {speech_time:.2f} seconds. Results saved to {speech_output_dir}")

    print(f"\nStarting video preprocessing for SONG videos from {VIDEO_SONG_PATH}...")
    start_time = time.time()
    song_features, song_metadata = process_video_dataset(
        VIDEO_SONG_PATH,
        song_output_dir,
        sample_rate=5,  
        visualize_subset=True 
    )
    song_time = time.time() - start_time
    print(f"Song video preprocessing completed in {song_time:.2f} seconds. Results saved to {song_output_dir}")

    total_videos = len(speech_metadata) + len(song_metadata)
    print(f"\nVideo preprocessing complete! Processed {total_videos} videos:")
    print(f"- Speech videos: {len(speech_metadata)}")
    print(f"- Song videos: {len(song_metadata)}")
























