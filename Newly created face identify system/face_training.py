import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import face_recognition
from PIL import Image, ImageEnhance, ImageFilter
import sqlite3
from datetime import datetime
from skimage import exposure, restoration, filters
import warnings
warnings.filterwarnings('ignore')

def advanced_image_enhancement(image_path):
    """
    Advanced image enhancement for very low quality images
    """
    try:
        # Read image with OpenCV for better handling
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            return None
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # 1. Noise reduction using Non-local Means Denoising
        img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
        
        # 2. Histogram equalization for better contrast
        img_yuv = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_eq, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 4. Sharpening using unsharp mask
        gaussian = cv2.GaussianBlur(img_clahe, (0, 0), 2.0)
        img_sharp = cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)
        
        # 5. Additional PIL enhancements
        pil_img = Image.fromarray(img_sharp)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(2.5)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.3)
        
        # Enhance brightness if too dark
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Apply unsharp mask filter
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # 6. Resize if too small (face_recognition works better with larger images)
        width, height = pil_img.size
        if width < 300 or height < 300:
            # Calculate new size maintaining aspect ratio
            if width < height:
                new_width = 400
                new_height = int((height * new_width) / width)
            else:
                new_height = 400
                new_width = int((width * new_height) / height)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        
        return np.array(pil_img)
        
    except Exception as e:
        print(f"Error enhancing image {image_path}: {e}")
        # Fallback to basic enhancement
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Basic enhancements
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            return np.array(img)
        except:
            return None

def detect_faces_multiple_methods(image):
    """
    Try multiple face detection methods for better results with low quality images
    """
    face_locations = []
    
    try:
        # Method 1: HOG (faster, good for frontal faces)
        face_locations = face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=2)
        
        # Method 2: If no faces found, try CNN (more accurate but slower)
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(image, model='cnn', number_of_times_to_upsample=1)
        
        # Method 3: If still no faces, try with different upsampling
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=3)
        
        # Method 4: Try OpenCV Haar Cascade as fallback
        if len(face_locations) == 0:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            opencv_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Convert OpenCV format to face_recognition format
            for (x, y, w, h) in opencv_faces:
                # face_recognition uses (top, right, bottom, left) format
                face_locations.append((y, x + w, y + h, x))
    
    except Exception as e:
        print(f"Face detection error: {e}")
    
    return face_locations

def normalize_face_encoding(encoding):
    """
    Enhanced normalization with multiple techniques
    """
    # L2 normalization
    norm = np.linalg.norm(encoding)
    if norm == 0:
        return encoding
    normalized = encoding / norm
    
    # Additional normalization: mean centering
    normalized = normalized - np.mean(normalized)
    
    # Re-normalize after mean centering
    norm = np.linalg.norm(normalized)
    if norm != 0:
        normalized = normalized / norm
    
    return normalized

def create_database():
    """
    Create SQLite database for storing face embeddings
    """
    conn = sqlite3.connect('face_embeddings.db')
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image_path TEXT,
            image_quality_score REAL,
            detection_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

def calculate_image_quality_score(image):
    """
    Calculate a simple quality score for the image
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate Laplacian variance (sharpness measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Normalize scores
        sharpness_score = min(laplacian_var / 1000, 1.0)  # Normalize to 0-1
        brightness_score = 1.0 - abs(brightness - 127) / 127  # Optimal around 127
        
        # Combined quality score
        quality_score = (sharpness_score + brightness_score) / 2
        return quality_score
    
    except:
        return 0.5  # Default score

def process_training_images(base_folder="recognize_images"):
    """
    Process all images in the training folder with enhanced preprocessing
    """
    if not os.path.exists(base_folder):
        print(f"Error: Folder '{base_folder}' not found!")
        return
    
    # Create database connection
    conn = create_database()
    cursor = conn.cursor()
    
    # Clear existing data (optional)
    cursor.execute("DELETE FROM face_embeddings")
    conn.commit()
    
    total_processed = 0
    total_failed = 0
    
    # Process each person's folder
    for person_name in os.listdir(base_folder):
        person_folder = os.path.join(base_folder, person_name)
        
        if not os.path.isdir(person_folder):
            continue
            
        print(f"Processing images for: {person_name}")
        person_processed = 0
        
        # Process each image in person's folder
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            
            # Check if it's an image file
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                continue
            
            try:
                print(f"  Processing {image_file}...")
                
                # Enhanced image preprocessing
                enhanced_image = advanced_image_enhancement(image_path)
                if enhanced_image is None:
                    print(f"    ✗ Could not enhance image")
                    total_failed += 1
                    continue
                
                # Calculate quality score
                quality_score = calculate_image_quality_score(enhanced_image)
                
                # Multiple face detection methods
                face_locations = detect_faces_multiple_methods(enhanced_image)
                
                if len(face_locations) == 0:
                    print(f"    ✗ No face found (tried multiple methods)")
                    total_failed += 1
                    continue
                
                # Get face encodings with better parameters
                try:
                    face_encodings = face_recognition.face_encodings(
                        enhanced_image, 
                        face_locations, 
                        num_jitters=10,  # More jitters for better encoding
                        model='large'    # Use large model for better accuracy
                    )
                except:
                    # Fallback to default parameters
                    face_encodings = face_recognition.face_encodings(enhanced_image, face_locations)
                
                if len(face_encodings) == 0:
                    print(f"    ✗ Could not encode face")
                    total_failed += 1
                    continue
                
                # Process each face found
                for idx, face_encoding in enumerate(face_encodings):
                    # Enhanced normalization
                    normalized_encoding = normalize_face_encoding(face_encoding)
                    
                    # Calculate detection confidence (based on face size)
                    face_location = face_locations[idx]
                    top, right, bottom, left = face_location
                    face_area = (bottom - top) * (right - left)
                    detection_confidence = min(face_area / 10000, 1.0)  # Normalize
                    
                    # Convert to bytes for storage
                    encoding_blob = pickle.dumps(normalized_encoding)
                    
                    # Store in database
                    cursor.execute('''
                        INSERT INTO face_embeddings 
                        (person_name, embedding, image_path, image_quality_score, detection_confidence)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (person_name, encoding_blob, image_path, quality_score, detection_confidence))
                    
                    person_processed += 1
                    total_processed += 1
                    print(f"    ✓ Processed (Quality: {quality_score:.2f}, Confidence: {detection_confidence:.2f})")
                
            except Exception as e:
                print(f"    ✗ Error processing {image_file}: {e}")
                total_failed += 1
        
        print(f"  Total processed for {person_name}: {person_processed}")
        print()
    
    # Commit all changes
    conn.commit()
    conn.close()
    
    print(f"Training completed!")
    print(f"Total images processed: {total_processed}")
    print(f"Total images failed: {total_failed}")
    print(f"Database saved as: face_embeddings.db")

def view_database_stats():
    """
    Display enhanced statistics about the stored embeddings
    """
    try:
        conn = sqlite3.connect('face_embeddings.db')
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        total_count = cursor.fetchone()[0]
        
        # Get count by person with quality stats
        cursor.execute("""
            SELECT person_name, COUNT(*), 
                   AVG(image_quality_score), AVG(detection_confidence)
            FROM face_embeddings 
            GROUP BY person_name
        """)
        person_stats = cursor.fetchall()
        
        print(f"Database Statistics:")
        print(f"Total embeddings stored: {total_count}")
        print(f"People in database:")
        for person, count, avg_quality, avg_confidence in person_stats:
            print(f"  - {person}: {count} images (Avg Quality: {avg_quality:.2f}, Avg Confidence: {avg_confidence:.2f})")
        
        conn.close()
        
    except Exception as e:
        print(f"Error reading database: {e}")

if __name__ == "__main__":
    print("Enhanced Face Recognition Training System")
    print("=" * 50)
    
    # Process training images
    process_training_images()
    
    # Show database statistics
    print("\n" + "=" * 50)
    view_database_stats()