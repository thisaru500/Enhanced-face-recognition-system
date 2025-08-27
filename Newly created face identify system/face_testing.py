import os
import cv2
import numpy as np
import pickle
import face_recognition
from PIL import Image, ImageEnhance, ImageFilter
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
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
        # Method 1: HOG with higher upsampling
        face_locations = face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=2)
        
        # Method 2: If no faces found, try CNN (more accurate but slower)
        if len(face_locations) == 0:
            try:
                face_locations = face_recognition.face_locations(image, model='cnn', number_of_times_to_upsample=1)
            except:
                pass  # CNN might not be available
        
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
        
        # Method 5: Try with even more aggressive upsampling as last resort
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(image, model='hog', number_of_times_to_upsample=4)
    
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

def load_database_embeddings():
    """
    Load all embeddings from database with quality filtering
    """
    try:
        conn = sqlite3.connect('face_embeddings.db')
        cursor = conn.cursor()
        
        # Load embeddings with quality and confidence scores
        cursor.execute("""
            SELECT id, person_name, embedding, image_quality_score, detection_confidence 
            FROM face_embeddings
            ORDER BY image_quality_score DESC, detection_confidence DESC
        """)
        results = cursor.fetchall()
        
        database_embeddings = []
        for row in results:
            person_id, person_name, embedding_blob, quality_score, confidence_score = row
            embedding = pickle.loads(embedding_blob)
            database_embeddings.append({
                'id': person_id,
                'name': person_name,
                'embedding': embedding,
                'quality_score': quality_score if quality_score else 0.5,
                'confidence_score': confidence_score if confidence_score else 0.5
            })
        
        conn.close()
        return database_embeddings
        
    except Exception as e:
        print(f"Error loading database: {e}")
        return []

def find_best_match_advanced(test_encoding, database_embeddings, base_threshold=0.5):
    """
    Advanced matching with multiple similarity metrics and weighted scoring
    """
    if not database_embeddings:
        return None, 0, {}
    
    candidates = []
    
    for db_entry in database_embeddings:
        # Calculate multiple similarity metrics
        cosine_sim = cosine_similarity([test_encoding], [db_entry['embedding']])[0][0]
        
        # Calculate Euclidean distance (converted to similarity)
        euclidean_dist = np.linalg.norm(test_encoding - db_entry['embedding'])
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Calculate dot product similarity
        dot_sim = np.dot(test_encoding, db_entry['embedding'])
        
        # Weighted combination of similarities
        quality_weight = db_entry['quality_score']
        confidence_weight = db_entry['confidence_score']
        
        # Combined similarity with weights
        combined_similarity = (
            cosine_sim * 0.5 +           # Primary metric
            euclidean_sim * 0.3 +        # Secondary metric
            dot_sim * 0.2                # Tertiary metric
        )
        
        # Apply quality and confidence weights
        weighted_similarity = combined_similarity * (0.7 + 0.15 * quality_weight + 0.15 * confidence_weight)
        
        # Dynamic threshold based on image quality
        dynamic_threshold = base_threshold * (0.8 + 0.2 * quality_weight)
        
        if weighted_similarity > dynamic_threshold:
            candidates.append({
                'entry': db_entry,
                'similarity': weighted_similarity,
                'cosine_sim': cosine_sim,
                'euclidean_sim': euclidean_sim,
                'dot_sim': dot_sim,
                'threshold_used': dynamic_threshold
            })
    
    if not candidates:
        return None, 0, {}
    
    # Sort by similarity and get the best match
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    best_candidate = candidates[0]
    
    # Additional validation: check if the best match is significantly better than others
    if len(candidates) > 1:
        second_best = candidates[1]['similarity']
        confidence_gap = best_candidate['similarity'] - second_best
        
        # If the gap is too small, reduce confidence
        if confidence_gap < 0.1:
            best_candidate['similarity'] *= 0.9
    
    return best_candidate['entry'], best_candidate['similarity'], {
        'cosine': best_candidate['cosine_sim'],
        'euclidean': best_candidate['euclidean_sim'],
        'dot_product': best_candidate['dot_sim'],
        'threshold': best_candidate['threshold_used'],
        'candidates_found': len(candidates)
    }

def process_test_image(image_path, database_embeddings):
    """
    Process a single test image with enhanced matching
    """
    print(f"Processing: {os.path.basename(image_path)}")
    
    try:
        # Enhanced image preprocessing
        enhanced_image = advanced_image_enhancement(image_path)
        if enhanced_image is None:
            print("  ✗ Error: Could not enhance image")
            return
        
        # Multiple face detection methods
        face_locations = detect_faces_multiple_methods(enhanced_image)
        
        if len(face_locations) == 0:
            print("  ✗ No face detected in image (tried multiple methods)")
            return
        
        print(f"  ✓ Found {len(face_locations)} face(s)")
        
        # Get face encodings with enhanced parameters
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
            print("  ✗ Could not encode faces in image")
            return
        
        # Process each face found
        for i, face_encoding in enumerate(face_encodings):
            print(f"  \nAnalyzing Face #{i+1}:")
            
            # Enhanced normalization
            normalized_encoding = normalize_face_encoding(face_encoding)
            
            # Advanced matching
            best_match, similarity, match_details = find_best_match_advanced(
                normalized_encoding, database_embeddings
            )
            
            if best_match:
                print(f"    ✅ MATCH FOUND!")
                print(f"       Person: {best_match['name']}")
                print(f"       ID: {best_match['id']}")
                print(f"       Overall Similarity: {similarity:.3f}")
                print(f"       Cosine Similarity: {match_details['cosine']:.3f}")
                print(f"       Euclidean Similarity: {match_details['euclidean']:.3f}")
                print(f"       Dot Product Similarity: {match_details['dot_product']:.3f}")
                print(f"       Quality Score: {best_match['quality_score']:.3f}")
                print(f"       Confidence Score: {best_match['confidence_score']:.3f}")
                print(f"       Threshold Used: {match_details['threshold']:.3f}")
                print(f"       Candidates Found: {match_details['candidates_found']}")
                print(f"    🎯 This photo is matching with ID {best_match['id']} and name {best_match['name']}")
                
                # Confidence level interpretation
                if similarity > 0.8:
                    print(f"    🟢 High Confidence Match")
                elif similarity > 0.65:
                    print(f"    🟡 Medium Confidence Match")
                else:
                    print(f"    🟠 Low Confidence Match")
            else:
                print(f"    ❌ No match found")
                print(f"       (No faces met the similarity threshold)")
            
        print()
    
    except Exception as e:
        print(f"  ✗ Error processing image: {e}")

def process_testing_folder(test_folder="testing_images"):
    """
    Process all images in the testing folder with enhanced analysis
    """
    if not os.path.exists(test_folder):
        print(f"Error: Testing folder '{test_folder}' not found!")
        return
    
    # Load database embeddings
    print("Loading database embeddings...")
    database_embeddings = load_database_embeddings()
    
    if not database_embeddings:
        print("Error: No embeddings found in database!")
        print("Please run the training script first.")
        return
    
    print(f"Loaded {len(database_embeddings)} embeddings from database")
    
    # Show quality statistics
    avg_quality = np.mean([entry['quality_score'] for entry in database_embeddings])
    avg_confidence = np.mean([entry['confidence_score'] for entry in database_embeddings])
    print(f"Database Average Quality: {avg_quality:.3f}")
    print(f"Database Average Confidence: {avg_confidence:.3f}")
    print("=" * 60)
    
    # Get all image files in testing folder
    image_files = []
    for file in os.listdir(test_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            image_files.append(os.path.join(test_folder, file))
    
    if not image_files:
        print("No image files found in testing folder!")
        return
    
    print(f"Found {len(image_files)} test images")
    print("=" * 60)
    
    # Process each test image
    successful_matches = 0
    total_faces_detected = 0
    
    for image_path in image_files:
        try:
            # Count faces before processing for statistics
            temp_img = advanced_image_enhancement(image_path)
            if temp_img is not None:
                temp_faces = detect_faces_multiple_methods(temp_img)
                total_faces_detected += len(temp_faces)
            
            process_test_image(image_path, database_embeddings)
        except Exception as e:
            print(f"Error with {image_path}: {e}")
    
    print("=" * 60)
    print(f"Testing Summary:")
    print(f"Images processed: {len(image_files)}")
    print(f"Total faces detected: {total_faces_detected}")

def test_single_image(image_path):
    """
    Test a single specific image with detailed analysis
    """
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found!")
        return
    
    # Load database embeddings
    print("Loading database embeddings...")
    database_embeddings = load_database_embeddings()
    
    if not database_embeddings:
        print("Error: No embeddings found in database!")
        return
    
    print(f"Loaded {len(database_embeddings)} embeddings from database")
    print("=" * 60)
    
    # Process the single image
    process_test_image(image_path, database_embeddings)

def interactive_testing():
    """
    Interactive testing mode where user can enter image paths
    """
    print("Enhanced Face Recognition Testing System")
    print("=" * 50)
    print("Interactive Mode - Enter image paths to test")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 50)
    
    # Load database embeddings once
    print("Loading database embeddings...")
    database_embeddings = load_database_embeddings()
    
    if not database_embeddings:
        print("Error: No embeddings found in database!")
        print("Please run the training script first.")
        return
    
    print(f"✓ Loaded {len(database_embeddings)} embeddings from database")
    print()
    
    while True:
        try:
            image_path = input("Enter image path (or 'quit' to exit): ").strip()
            
            if image_path.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not image_path:
                continue
            
            print("=" * 60)
            process_test_image(image_path, database_embeddings)
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def batch_test_with_results():
    """
    Batch testing with detailed results and statistics
    """
    test_folder = "testing_images"
    if not os.path.exists(test_folder):
        print(f"Error: Testing folder '{test_folder}' not found!")
        return
    
    # Load database embeddings
    print("Loading database embeddings...")
    database_embeddings = load_database_embeddings()
    
    if not database_embeddings:
        print("Error: No embeddings found in database!")
        return
    
    print(f"Loaded {len(database_embeddings)} embeddings from database")
    
    # Collect all results
    results = []
    image_files = [f for f in os.listdir(test_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))]
    
    print(f"\nProcessing {len(image_files)} images...")
    print("=" * 60)
    
    for image_file in image_files:
        image_path = os.path.join(test_folder, image_file)
        print(f"Processing: {image_file}")
        
        try:
            # Enhanced image preprocessing
            enhanced_image = advanced_image_enhancement(image_path)
            if enhanced_image is None:
                results.append({'image': image_file, 'status': 'Enhancement Failed'})
                continue
            
            # Face detection
            face_locations = detect_faces_multiple_methods(enhanced_image)
            if len(face_locations) == 0:
                results.append({'image': image_file, 'status': 'No Face Detected'})
                continue
            
            # Face encoding
            try:
                face_encodings = face_recognition.face_encodings(
                    enhanced_image, face_locations, num_jitters=10, model='large'
                )
            except:
                face_encodings = face_recognition.face_encodings(enhanced_image, face_locations)
            
            if len(face_encodings) == 0:
                results.append({'image': image_file, 'status': 'Encoding Failed'})
                continue
            
            # Process first face
            face_encoding = face_encodings[0]
            normalized_encoding = normalize_face_encoding(face_encoding)
            
            # Find match
            best_match, similarity, match_details = find_best_match_advanced(
                normalized_encoding, database_embeddings
            )
            
            if best_match:
                results.append({
                    'image': image_file,
                    'status': 'Match Found',
                    'person': best_match['name'],
                    'id': best_match['id'],
                    'similarity': similarity,
                    'cosine_sim': match_details['cosine'],
                    'quality': best_match['quality_score']
                })
                print(f"  ✅ Matched: {best_match['name']} (Similarity: {similarity:.3f})")
            else:
                results.append({'image': image_file, 'status': 'No Match Found'})
                print(f"  ❌ No match found")
                
        except Exception as e:
            results.append({'image': image_file, 'status': f'Error: {str(e)}'})
            print(f"  ✗ Error: {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH TESTING RESULTS SUMMARY")
    print("=" * 60)
    
    matches = [r for r in results if r['status'] == 'Match Found']
    no_matches = [r for r in results if r['status'] == 'No Match Found']
    errors = [r for r in results if r['status'] not in ['Match Found', 'No Match Found']]
    
    print(f"Total Images Processed: {len(results)}")
    print(f"Successful Matches: {len(matches)}")
    print(f"No Matches Found: {len(no_matches)}")
    print(f"Processing Errors: {len(errors)}")
    print(f"Success Rate: {len(matches)/len(results)*100:.1f}%")
    
    if matches:
        print(f"\nMatch Details:")
        for match in matches:
            print(f"  {match['image']} -> {match['person']} (ID: {match['id']}, Sim: {match['similarity']:.3f})")
    
    if errors:
        print(f"\nErrors:")
        for error in errors:
            print(f"  {error['image']}: {error['status']}")

def view_database_info():
    """
    Display detailed information about the database
    """
    try:
        conn = sqlite3.connect('face_embeddings.db')
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        total_count = cursor.fetchone()[0]
        
        # Get count by person with detailed stats
        cursor.execute("""
            SELECT person_name, COUNT(*), 
                   AVG(image_quality_score), AVG(detection_confidence),
                   MIN(image_quality_score), MAX(image_quality_score)
            FROM face_embeddings 
            GROUP BY person_name
            ORDER BY COUNT(*) DESC
        """)
        person_stats = cursor.fetchall()
        
        print("DATABASE INFORMATION")
        print("=" * 50)
        print(f"Total embeddings: {total_count}")
        print(f"Unique people: {len(person_stats)}")
        print()
        print("Person Statistics:")
        print("-" * 80)
        print("Name".ljust(20) + "Count".ljust(8) + "Avg Quality".ljust(12) + 
              "Avg Confidence".ljust(15) + "Quality Range")
        print("-" * 80)
        
        for person, count, avg_quality, avg_confidence, min_quality, max_quality in person_stats:
            print(f"{person[:19].ljust(20)}{str(count).ljust(8)}{avg_quality:.3f}".ljust(12) + 
                  f"{avg_confidence:.3f}".ljust(15) + f"{min_quality:.3f} - {max_quality:.3f}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error reading database: {e}")

def main_menu():
    """
    Main menu for the testing system
    """
    while True:
        print("\n" + "=" * 50)
        print("ENHANCED FACE RECOGNITION TESTING SYSTEM")
        print("=" * 50)
        print("1. Test all images in 'testing_images' folder")
        print("2. Test a single image")
        print("3. Interactive testing mode")
        print("4. Batch test with detailed results")
        print("5. View database information")
        print("6. Exit")
        print("=" * 50)
        
        try:
            choice = input("Select an option (1-6): ").strip()
            
            if choice == '1':
                process_testing_folder()
            elif choice == '2':
                image_path = input("Enter image path: ").strip()
                test_single_image(image_path)
            elif choice == '3':
                interactive_testing()
            elif choice == '4':
                batch_test_with_results()
            elif choice == '5':
                view_database_info()
            elif choice == '6':
                print("Goodbye!")
                break
            else:
                print("Invalid choice! Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Check if database exists
    if not os.path.exists('face_embeddings.db'):
        print("Error: face_embeddings.db not found!")
        print("Please run the training script first to create the database.")
    else:
        main_menu()