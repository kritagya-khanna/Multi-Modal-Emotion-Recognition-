import os 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Add this import for visualization
from pathlib import Path
from tqdm import tqdm
import dlib
import time 
import warnings 
import math
from scipy.spatial import distance
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # Add this for analysis function
from skimage.feature import hog 
from skimage.color import rgb2gray
warnings.filterwarnings("ignore")

from data_setup import EMOTION_MAP, PROJECT_ROOT, DATA_PATH, PROCESSED_PATH

print("initializing video preprocessing...")

# initialize dlib's face detector 
face_detector = dlib.get_frontal_face_detector()

SHAPE_PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"

try:
    landmark_predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
    print("Landmark predictor loaded successfully.")
except Exception as e:
    print(f"Error loading landmark predictor: {e}")
    print(f"Looking for file at: {SHAPE_PREDICTOR_PATH}")
    print("Please ensure the shape predictor file is available.")
    exit(1)

print("initializing video preprocessing done.")

def extract_frames(video_path, sample_rate = 5, max_frames = None):
    # Extract frames from a video file at a specific sample rate
    # video_path (str): Path to the video file
    # sample_rate (int): Extract every Nth frame
    # max_frames (int, optional): Maximum number of frames to extract

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open video file: {video_path}")
            return [], 0, {'path': video_path, 'filename': os.path.basename(video_path), 'error': 'Could not open file'}

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

        # Extract metadata from filename (RAVDESS format)
        filename = video_metadata['filename']
        parts = filename.split('-')
        
        if len(parts) >= 7:
            try:
                video_metadata['modality'] = parts[0]  # This should be 01 or 02
                video_metadata['vocal_channel'] = parts[1]  # This should be 01 or 02 
                video_metadata['emotion_code'] = parts[2]  # This contains the emotion code (01-08)
                video_metadata['emotion'] = EMOTION_MAP.get(parts[2], 'unknown')  # Map the emotion code to name
                video_metadata['intensity'] = 'normal' if parts[3] == '01' else 'strong'
                video_metadata['statement'] = parts[4]
                video_metadata['repetition'] = parts[5]
                actor_id = parts[6].split('.')[0]
                video_metadata['actor_id'] = actor_id
                video_metadata['actor'] = actor_id
                video_metadata['gender'] = 'female' if int(actor_id) % 2 == 0 else 'male'
            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                # Add default values
                video_metadata.setdefault('modality', 'unknown')
                video_metadata.setdefault('vocal_channel', 'unknown')
                video_metadata.setdefault('emotion', 'unknown')
                video_metadata.setdefault('emotion_code', 'unknown')
                video_metadata.setdefault('intensity', 'unknown')
                video_metadata.setdefault('statement', 'unknown')
                video_metadata.setdefault('repetition', 'unknown')
                video_metadata.setdefault('actor_id', 'unknown')
                video_metadata.setdefault('actor', 'unknown')
                video_metadata.setdefault('gender', 'unknown')
        else:
            print(f"Warning: Filename format not recognized: {filename}")
            video_metadata['emotion'] = 'unknown'
            video_metadata['actor_id'] = 'unknown'
            video_metadata['actor'] = 'unknown'
            video_metadata['gender'] = 'unknown'
            video_metadata['intensity'] = 'unknown'

        frames = []
        frame_indices = []
        frame_index = 0


        # Calculate total number of frames we'll be extracting (not total video frames)
        expected_frames = min(frame_count // frame_interval + (1 if frame_count % frame_interval > 0 else 0), 
                             max_frames if max_frames else float('inf'))
        

        # Show progress bar for frame extraction
        pbar = tqdm(total=expected_frames, 
                    desc=f"Extracting frames from {os.path.basename(video_path)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(frame_index)
                pbar.update(1)

                if max_frames is not None and len(frames) >= max_frames:
                    break

            frame_index += 1

        pbar.close()
        cap.release()

        video_metadata['extracted_frames'] = len(frames)
        video_metadata['frame_indices'] = frame_indices

        print(f"Extracted {len(frames)} frames from {video_path} at a sample rate of {sample_rate}.")
        return frames, frame_count, video_metadata
    
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return [], 0, {'path': video_path, 'filename': os.path.basename(video_path), 'error': str(e)}


def detect_faces_landmarks(frame):
    # Detect faces and landmarks in a single frame using Dlib
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)

        if len(faces) == 0:
            return False, None, None, 0.0
        
        face = faces[0]  # Use the first face detected
        detection_confidence = 1.0

        # Get the landmarks for the face
        face_bbox = (face.left(), face.top(), face.right(), face.bottom())
        shape = landmark_predictor(gray, face)

        # Convert landmarks to a list of (x, y, z) tuples
        h, w = gray.shape
        landmarks = []

        for i in range(68):
            x = shape.part(i).x / w  # normalize to [0, 1]
            y = shape.part(i).y / h
            landmarks.append((x, y, 0.0))  # z is set to 0.0

        return True, landmarks, face_bbox, detection_confidence
    
    except Exception as e:
        print(f"Error in face detection: {e}")
        return False, None, None, 0.0


def align_face(frame, landmarks, bbox, target_size=(224, 224)):
    # Align the face in the frame using landmarks and bounding box
    try:
        x_min, y_min, x_max, y_max = bbox

        # Add margin around face
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
            
            # Enhance image using CLAHE
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced_face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_face
        else:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    except Exception as e:
        print(f"Error in face alignment: {e}")
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)


def extract_geometric_features(landmarks):
    # Extract geometric features from landmarks
    try:
        features = {}
        landmarks_array = np.array(landmarks)

        # Define facial feature regions
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

        # 1. Eye aspect ratio (EAR) - measures eye openness
        def eye_aspect_ratio(eye_landmarks):
            # Vertical distances
            vert1 = distance.euclidean(landmarks[eye_landmarks[1]], landmarks[eye_landmarks[5]])
            vert2 = distance.euclidean(landmarks[eye_landmarks[2]], landmarks[eye_landmarks[4]])
            # Horizontal distance
            horiz = distance.euclidean(landmarks[eye_landmarks[0]], landmarks[eye_landmarks[3]])
            # Calculate EAR
            ear = (vert1 + vert2) / (2.0 * horiz) if horiz > 0 else 0
            return ear
        
        features['left_eye_ear'] = eye_aspect_ratio(left_eye)
        features['right_eye_ear'] = eye_aspect_ratio(right_eye)
        features['avg_eye_ear'] = (features['left_eye_ear'] + features['right_eye_ear']) / 2.0

        # 2. Mouth aspect ratio (MAR) - measures mouth openness
        mouth_top = landmarks_array[mouth_outline[3]]  # Upper lip
        mouth_bottom = landmarks_array[mouth_outline[9]]  # Lower lip
        mouth_left = landmarks_array[mouth_outline[0]]  # Left corner
        mouth_right = landmarks_array[mouth_outline[6]]  # Right corner

        mouth_width = distance.euclidean(mouth_left, mouth_right)
        mouth_height = distance.euclidean(mouth_top, mouth_bottom)

        features['mouth_aspect_ratio'] = mouth_height / mouth_width if mouth_width > 0 else 0

        # 3. Eyebrow position relative to eyes
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
    
    except Exception as e:
        print(f"Error extracting geometric features: {e}")
        return {}


def extract_appearance_features(aligned_face, hog_orientation=9, hog_pixel_per_cell=(8, 8), hog_cells_per_block=(2, 2)):
    # Extract appearance features from the aligned face
    try:
        features = {}

        gray_face = rgb2gray(aligned_face)

        # Compute HOG features
        hog_features, hog_image = hog(
            gray_face, 
            orientations=hog_orientation, 
            pixels_per_cell=hog_pixel_per_cell, 
            cells_per_block=hog_cells_per_block, 
            visualize=True, 
            block_norm="L2-Hys"
        )
        
        features['hog_mean'] = np.mean(hog_features)
        features['hog_std'] = np.std(hog_features)
        features['hog_min'] = np.min(hog_features)
        features['hog_max'] = np.max(hog_features)
        
        # Basic image statistics
        features['face_brightness'] = np.mean(gray_face)
        features['face_contrast'] = np.std(gray_face)
        
        # Edge intensity
        sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features['edge_intensity_mean'] = np.mean(edge_magnitude)
        features['edge_intensity_std'] = np.std(edge_magnitude)
        
        return features, hog_image
    
    except Exception as e:
        print(f"Error extracting appearance features: {e}")
        return {}, None


def draw_landmarks(frame, landmarks, bbox):
    # Draw facial landmarks and bounding box on the frame
    try:
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
    
    except Exception as e:
        print(f"Error drawing landmarks: {e}")
        return frame


def process_video(video_path, output_dir, visualize=False, sample_rate=5, max_frame=None):
    """Process a single video file: extract frames, detect faces, extract features
    
    Args:
        video_path: Path to the video file
        output_dir: Path to save results
        visualize: Whether to generate visualizations
        sample_rate: Extract every Nth frame
        max_frame: Maximum number of frames to process
        
    Returns:
        list of feature dictionaries, video metadata
    """
    try:
        frames, frame_count, video_metadata = extract_frames(video_path, sample_rate, max_frame)
        
        if not frames or len(frames) == 0:
            print(f"No frames extracted from {video_path}.")
            return None, video_metadata
        
        all_features = []
        face_detected_count = 0
        
        if visualize:
            vid_name = os.path.basename(video_path).split('.')[0]
            vis_dir = Path(output_dir) / "visualization" / vid_name
            os.makedirs(vis_dir, exist_ok=True)
        
        for i, frame in enumerate(tqdm(frames, desc=f"Processing {os.path.basename(video_path)}")):
            frame_features = {
                'frame_index': video_metadata['frame_indices'][i],
                'timestamp': video_metadata['frame_indices'][i] / video_metadata['fps'] if video_metadata['fps'] > 0 else 0
            }
            
            try:
                landmarks_detected, landmarks, bbox, confidence = detect_faces_landmarks(frame)
                
                if landmarks_detected:
                    face_detected_count += 1
                    frame_features['face_detected'] = True
                    frame_features['detection_confidence'] = confidence
                    
                    # Process detected face
                    aligned_face = align_face(frame, landmarks, bbox)
                    
                    # Extract features
                    geometric_features = extract_geometric_features(landmarks)
                    frame_features.update(geometric_features)
                    
                    appearance_features, hog_image = extract_appearance_features(aligned_face)
                    frame_features.update(appearance_features)
                    
                    # Generate visualizations if requested
                    if visualize and i % 10 == 0:
                        vis_frame = draw_landmarks(frame, landmarks, bbox)
                        
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
                        
                        # Key features
                        plt.subplot(2, 2, 4)
                        features_to_plot = {
                            'Mouth AR': geometric_features['mouth_aspect_ratio'],
                            'Eye AR': geometric_features['avg_eye_ear'],
                            'Smile': geometric_features['smile_ratio'],
                            'Eyebrow': geometric_features['left_eyebrow_eye_dist']
                        }
                        plt.bar(features_to_plot.keys(), features_to_plot.values())
                        plt.title(f'Key Features - {video_metadata["emotion"]}')
                        plt.tight_layout()
                        
                        # Save visualization
                        plt.savefig(vis_dir / f"frame_{i:04d}.png")
                        plt.close()
                
                else:
                    frame_features['face_detected'] = False
                    frame_features['detection_confidence'] = 0
            
            except Exception as e:
                print(f"Error processing frame {i} from {video_path}: {e}")
                frame_features['face_detected'] = False
                frame_features['detection_confidence'] = 0
            
            # Add metadata from video to each frame
            for key in ['emotion', 'emotion_code', 'actor_id', 'actor', 'gender', 'intensity']:
                if key in video_metadata:
                    frame_features[key] = video_metadata[key]
            
            all_features.append(frame_features)
        
        # Update metadata with face detection statistics
        video_metadata['face_detected_count'] = face_detected_count
        video_metadata['face_detection_rate'] = face_detected_count / len(frames) if frames else 0
        
        return all_features, video_metadata
    
    except Exception as e:
        print(f"Failed to process video {video_path}: {e}")
        return None, {'path': video_path, 'filename': os.path.basename(video_path), 'error': str(e)}


def analyze_video_features(features_df, metadata_df, output_path):
    """Analyze extracted video features and generate visualizations
    
    Args:
        features_df: DataFrame containing features
        metadata_df: DataFrame containing metadata
        output_path: path to save the visualizations
    """
    analysis_dir = Path(output_path) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Generating video feature analysis...")
        
        # Combine data for analysis
        combined_df = features_df.copy()
        
        # 1. Distribution of emotion classes
        plt.figure(figsize=(12, 6))
        emotion_counts = combined_df['emotion'].value_counts()
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        plt.title('Distribution of Emotions in Video Dataset')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(analysis_dir / "video_emotion_distribution.png")
        plt.close()
        
        # 2. Gender distribution
        plt.figure(figsize=(10, 5))
        gender_counts = combined_df['gender'].value_counts()
        sns.barplot(x=gender_counts.index, y=gender_counts.values)
        plt.title('Gender Distribution in Video Dataset')
        plt.tight_layout()
        plt.savefig(analysis_dir / "video_gender_distribution.png")
        plt.close()
        
        # 3. Feature distribution by emotion (for key features)
        key_features = ['mouth_aspect_ratio', 'avg_eye_ear', 'smile_ratio', 'face_brightness', 'hog_mean']
        
        for feature in key_features:
            if feature in features_df.columns:
                plt.figure(figsize=(12, 6))
                valid_data = combined_df[combined_df['face_detected'] == True]
                sns.boxplot(x='emotion', y=feature, data=valid_data)
                plt.title(f'Distribution of {feature} by Emotion')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(analysis_dir / f"video_{feature}_by_emotion.png")
                plt.close()
        
        # 4. Correlation matrix of features
        valid_df = features_df[features_df['face_detected'] == True].copy()
        if len(valid_df) > 0:
            numeric_features = valid_df.select_dtypes(include=[np.number])
            # Remove non-feature columns
            cols_to_exclude = ['frame_index', 'timestamp', 'video_id', 'face_detected', 'detection_confidence']
            feature_cols = [col for col in numeric_features.columns if col not in cols_to_exclude]
            
            if feature_cols:
                correlation_matrix = numeric_features[feature_cols].corr()
                
                plt.figure(figsize=(20, 16))
                sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, square=True)
                plt.title("Video Features Correlation Matrix")
                plt.tight_layout()
                plt.savefig(analysis_dir / "video_features_correlation_matrix.png")
                plt.close()
                
                # 5. PCA of features
                features_for_pca = numeric_features[feature_cols].fillna(0)
                
                if len(features_for_pca) > 2:  # Need at least 2 samples for PCA
                    # Scale the features before PCA
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(features_for_pca)
                    
                    pca = PCA(n_components=2)
                    principal_components = pca.fit_transform(scaled_features)
                    
                    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                    pca_df['emotion'] = valid_df['emotion'].values
                    
                    plt.figure(figsize=(12, 8))
                    sns.scatterplot(x='PC1', y='PC2', hue='emotion', data=pca_df)
                    plt.title("PCA of Video Features by Emotion")
                    plt.tight_layout()
                    plt.savefig(analysis_dir / "video_PCA_visualization.png")
                    plt.close()
        
        # Generate a summary text file
        with open(analysis_dir / 'video_feature_analysis.txt', 'w') as f:
            f.write("Video Feature Analysis Summary\n")
            f.write("============================\n\n")
            f.write(f"Total samples: {len(features_df)}\n")
            f.write(f"Face detection rate: {features_df['face_detected'].mean():.2f}\n")
            f.write(f"Emotion distribution:\n{emotion_counts.to_string()}\n\n")
            f.write(f"Gender distribution:\n{gender_counts.to_string()}\n\n")
            
            if len(valid_df) > 0:
                f.write(f"Feature statistics (only frames with detected faces):\n{valid_df.describe().to_string()}\n\n")
                
                if 'pca' in locals():
                    f.write(f"PCA explained variance ratio: {pca.explained_variance_ratio_}\n")
            
            f.write("\nTop features for emotion discrimination:\n")
            f.write("- mouth_aspect_ratio: related to mouth openness\n")
            f.write("- smile_ratio: related to smiling\n")
            f.write("- eye_openness_ratio: related to surprise/fear\n")
            f.write("- eyebrow_eye_dist: related to surprise/sadness\n")
        
        print(f"Video feature analysis complete. Results saved to {analysis_dir}")
    
    except Exception as e:
        print(f"Error during video feature analysis: {e}")


def process_video_dataset(video_dir, output_dir, sample_rate=5, max_videos=None, visualize_subset=True):
    """Process all videos in directory
    
    Args:
        video_dir: Path to directory containing videos
        output_dir: Path to save processed features
        sample_rate: Extract every Nth frame
        max_videos: Maximum number of videos to process (for testing)
        visualize_subset: Whether to visualize a subset of videos
    
    Returns:
        features_df: DataFrame containing features
        metadata_df: DataFrame containing metadata
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    video_extensions = ['.mp4', '.avi', '.mov']
    video_files = []
    
    # Find all video files with the specified extensions
    for ext in video_extensions:
        video_files.extend(list(Path(video_dir).rglob(f"*{ext}")))

    if not video_files:
        print(f"No video files found in {video_dir}")
        print(f"Looking for files with extensions: {video_extensions}")
        return pd.DataFrame(), pd.DataFrame()

    video_files.sort()

    if max_videos is not None:
        video_files = video_files[:max_videos]

    print(f"Found {len(video_files)} videos in {video_dir}")
    
    all_video_features = []
    all_metadata = []
    processed_count = 0

    for i, video_path in enumerate(video_files):
        vis_this_video = visualize_subset and i % 10 == 0
        
        print(f"\nProcessing video {i + 1}/{len(video_files)}: {video_path}")
        
        try:
            features_list, metadata = process_video(
                str(video_path), 
                output_dir, 
                visualize=vis_this_video, 
                sample_rate=sample_rate
            )
            
            if not features_list:
                print(f"No features extracted from {video_path}.")
                continue
                
            video_id = i
            for features in features_list:
                features['video_id'] = video_id
                all_video_features.append(features)
                
            metadata['video_id'] = video_id
            all_metadata.append(metadata)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue

    print(f"Successfully processed {processed_count} out of {len(video_files)} videos")
    
    # Create DataFrames for features and metadata
    features_df = pd.DataFrame(all_video_features)
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save to CSV
    if not features_df.empty:
        features_df.to_csv(output_dir / "video_features.csv", index=False)
        metadata_df.to_csv(output_dir / "video_metadata.csv", index=False)
        
        # Create combined file (like in audio processing)
        # Only include metadata columns not already in features
        metadata_cols = [col for col in metadata_df.columns if col not in features_df.columns]
        if metadata_cols:
            extended_features = features_df.copy()
            for col in metadata_cols:
                video_to_metadata = {meta['video_id']: meta[col] for _, meta in metadata_df.iterrows() if col in meta}
                extended_features[col] = extended_features['video_id'].map(video_to_metadata)
            extended_features.to_csv(output_dir / "video_combined.csv", index=False)
        
        # Process only frames with detected faces for scaling and modeling
        if 'face_detected' in features_df.columns:
            # Filter and standardize the features
            face_detected_df = features_df[features_df['face_detected'] == True].copy()
            
            if len(face_detected_df) > 0:
                # Extract numeric columns for scaling
                numeric_features = face_detected_df.select_dtypes(include=[np.number])
                cols_to_exclude = ['frame_index', 'timestamp', 'video_id', 'face_detected', 'detection_confidence']
                feature_cols = [col for col in numeric_features.columns if col not in cols_to_exclude]
                
                # Handle missing values before scaling
                features_for_scaling = numeric_features[feature_cols].fillna(0)
                
                # Scale the features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features_for_scaling)
                
                # Save the scaler for later use
                joblib.dump(scaler, output_dir / "video_feature_scaler.pkl")
                
                # Create DataFrame with scaled features
                scaled_df = pd.DataFrame(scaled_features, columns=feature_cols)
                
                # Add metadata columns to scaled features
                meta_cols = ['emotion', 'actor_id', 'gender', 'intensity', 'video_id']
                for col in meta_cols:
                    if col in face_detected_df.columns:
                        scaled_df[col] = face_detected_df[col].values
                
                # Save scaled features
                scaled_df.to_csv(output_dir / "scaled_video_features.csv", index=False)
                
                # Generate analysis of video features
                analyze_video_features(features_df, metadata_df, output_dir)
                
                # Create a balanced dataset for training
                emotion_groups = scaled_df.groupby('emotion')
                min_count = emotion_groups.size().min()
                balanced_df = pd.concat([group.sample(min_count) for name, group in emotion_groups])
                balanced_df.to_csv(output_dir / "balanced_video_features.csv", index=False)
                
                print(f"Created balanced dataset with {len(balanced_df)} samples per emotion")
    else:
        print("No features were extracted from any videos. Check your data or parameters.")
    
    print(f"Video preprocessing complete! Successfully processed {processed_count} videos.")
    print(f"Features and metadata saved to {output_dir}")
    
    return features_df, metadata_df


if __name__ == "__main__":
    from data_setup import DATA_PATH, PROCESSED_PATH
    import time

    # Check if data directories exist
    VIDEO_SPEECH_PATH = DATA_PATH / "Video_Speech_Actors_01-24"
    VIDEO_SONG_PATH = DATA_PATH / "Video_Song_Actors_01-24"
    
    if not VIDEO_SPEECH_PATH.exists():
        print(f"Warning: Speech video directory not found at {VIDEO_SPEECH_PATH}")
        print(f"Available directories in {DATA_PATH}:")
        for item in DATA_PATH.iterdir():
            if item.is_dir():
                print(f" - {item}")
    
    if not VIDEO_SONG_PATH.exists():
        print(f"Warning: Song video directory not found at {VIDEO_SONG_PATH}")
        print(f"Available directories in {DATA_PATH}:")
        for item in DATA_PATH.iterdir():
            if item.is_dir():
                print(f" - {item}")

    speech_output_dir = PROCESSED_PATH / "video_features" / "speech"
    song_output_dir = PROCESSED_PATH / "video_features" / "song"

    os.makedirs(speech_output_dir, exist_ok=True)
    os.makedirs(song_output_dir, exist_ok=True)

    # Process speech videos if directory exists
    speech_features, speech_metadata = None, None
    if VIDEO_SPEECH_PATH.exists():
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
    else:
        print("Skipping speech video processing as directory was not found")

    # Process song videos if directory exists
    song_features, song_metadata = None, None
    if VIDEO_SONG_PATH.exists():
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
    else:
        print("Skipping song video processing as directory was not found")

    # Summary
    speech_count = len(speech_metadata) if speech_metadata is not None else 0
    song_count = len(song_metadata) if song_metadata is not None else 0
    total_videos = speech_count + song_count
    
    print(f"\nVideo preprocessing complete! Processed {total_videos} videos:")
    print(f"- Speech videos: {speech_count}")
    print(f"- Song videos: {song_count}")
























