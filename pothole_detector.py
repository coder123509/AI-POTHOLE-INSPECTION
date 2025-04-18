import streamlit as st
import cv2
import torch
import tempfile
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from threading import Thread
import queue
import textwrap

st.set_page_config(
    page_title="RoadGuardian: AI Pothole Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #2d5f91;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424f5c;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #c0e6ae;
        color: #0e7526;
        padding: 1rem;
        border-radius: 0.65rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #7de6fa;
        color: #054f5c;
        padding: 1rem;
        border-radius: 0.65rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #e8f1fa;
        border-radius: 0.65rem;
        padding: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 1.2rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path, conf_threshold=0.25):
    """Load YOLOv5 model and cache it for performance"""
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
    model.conf = conf_threshold
    return model




@st.cache_resource
def load_model(model_path, conf_threshold=0.25):
    """Load YOLOv5 model and cache it for performance"""
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
    model.conf = conf_threshold
    return model
# Add this function at the appropriate location in your code


def process_image(model, image, conf_threshold=0.25):
    """Process a single image and return results"""
    model.conf = conf_threshold
    
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        # Convert RGB to BGR for OpenCV processing
        img = img_array[:, :, ::-1].copy()
    else:
        img = image
    
    # Run inference
    start_time = time.time()
    results = model(img)
    inference_time = time.time() - start_time
    
    # Get detection results
    detections = results.xyxy[0].cpu().numpy()
    
    # Render results on image
    result_img = results.render()[0]
    
    # If input was PIL, convert back to RGB for display
    if isinstance(image, Image.Image):
        result_img = result_img[:, :, ::-1]
    
    return result_img, detections, inference_time

def display_metrics(col, value, label, delta=None, delta_color="normal"):
    """Display a metric in a column with styling"""
    with col:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

def process_video(model, video_path, conf_threshold=0.25, output_path=None, progress_callback=None):
    """Process a video file and return statistics"""
    model.conf = conf_threshold
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video source")
        return None, None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer if output path is specified
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Variables to track statistics
    frame_count = 0
    pothole_counts = []
    processing_times = []
    total_potholes = 0
    frames_with_potholes = 0
    severity_levels = {
        "Low": 0,     # Small potholes
        "Medium": 0,  # Medium potholes
        "High": 0     # Large potholes
    }
    
    # Process the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        start_time = time.time()
        results = model(frame)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Get detections
        detections = results.xyxy[0].cpu().numpy()
        num_potholes = len(detections)
        pothole_counts.append(num_potholes)
        total_potholes += num_potholes
        
        if num_potholes > 0:
            frames_with_potholes += 1
            
            # Classify severity based on bounding box size
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                area = (x2 - x1) * (y2 - y1)
                area_percentage = area / (width * height)
                
                if area_percentage < 0.01:
                    severity_levels["Low"] += 1
                elif area_percentage < 0.05:
                    severity_levels["Medium"] += 1
                else:
                    severity_levels["High"] += 1
        
        # Render results
        annotated_frame = results.render()[0]
        
        # Add text with detection stats
        cv2.putText(annotated_frame, f"Potholes: {num_potholes}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"FPS: {1/max(processing_time, 0.001):.1f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write to output video
        writer.write(annotated_frame)
        
        # Update progress
        frame_count += 1
        if progress_callback and total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            status = f"Processing frame {frame_count}/{total_frames} | Potholes detected: {total_potholes}"
            progress_callback(progress, status)
    
    # Release resources
    cap.release()
    writer.release()
    
    # Prepare statistics
    stats = {
        'total_frames': frame_count,
        'frames_with_potholes': frames_with_potholes,
        'pothole_frame_percentage': (frames_with_potholes / max(frame_count, 1)) * 100,
        'total_potholes': total_potholes,
        'avg_potholes_per_frame': total_potholes / max(frame_count, 1),
        'avg_processing_time': sum(processing_times) / max(len(processing_times), 1),
        'avg_fps': 1 / max(sum(processing_times) / max(len(processing_times), 1), 0.001),
        'severity_levels': severity_levels,
        'pothole_counts': pothole_counts
    }
    
    return output_path, stats

def display_video_analytics(stats):
    """Display analytics for processed video"""
    st.subheader("📊 Detection Analytics")
    
    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    display_metrics(col1, stats['total_potholes'], "Total Potholes")
    display_metrics(col2, f"{stats['pothole_frame_percentage']:.1f}%", "Frames with Potholes")
    display_metrics(col3, f"{stats['avg_potholes_per_frame']:.2f}", "Avg. Potholes per Frame")
    display_metrics(col4, f"{stats['avg_fps']:.1f}", "Avg. FPS")
    
    # Create severity distribution chart
    st.subheader("Pothole Severity Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    severities = stats['severity_levels']
    ax.bar(['Low', 'Medium', 'High'], 
           [severities['Low'], severities['Medium'], severities['High']],
           color=['#3498db', '#f39c12', '#e74c3c'])
    ax.set_ylabel('Count')
    ax.set_title('Pothole Severity Distribution')
    st.pyplot(fig)
    
    # Create pothole count over time chart
    st.subheader("Pothole Detection Over Time")
    
    # Display only if the video has enough frames
    if len(stats['pothole_counts']) > 10:
        # Downsample if too many frames for readability
        if len(stats['pothole_counts']) > 100:
            step = len(stats['pothole_counts']) // 100
            x = list(range(0, len(stats['pothole_counts']), step))
            y = [stats['pothole_counts'][i] for i in x]
        else:
            x = list(range(len(stats['pothole_counts'])))
            y = stats['pothole_counts']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, y, color='#2ecc71', linewidth=2)
        ax.fill_between(x, y, color='#2ecc71', alpha=0.2)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Pothole Count')
        ax.set_title('Potholes Detected per Frame')
        st.pyplot(fig)
    else:
        st.info("Not enough frames to generate pothole timeline chart.")
    
    # Recommendations based on detections
    st.subheader("🚧 Road Condition Assessment")
    
    road_condition = "Good"
    if stats['total_potholes'] > 50 or severities['High'] > 10:
        road_condition = "Poor"
    elif stats['total_potholes'] > 20 or severities['Medium'] > 10:
        road_condition = "Fair"
    
    st.markdown(f"""
    <div class="info-box">
        <b>Road Condition:</b> {road_condition}<br>
        <b>Analysis:</b> This road segment contains {stats['total_potholes']} potholes 
        ({severities['Low']} low severity, {severities['Medium']} medium severity, 
        {severities['High']} high severity).<br>
        <b>Recommendation:</b> {
            "Immediate repair needed for major potholes." if road_condition == "Poor" else
            "Schedule repair for medium severity potholes." if road_condition == "Fair" else
            "Monitor for any new potholes forming."
        }
    </div>
    """, unsafe_allow_html=True)



def main():
    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="main-header">🛣️ RoadGuardian AI</div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center;">Advanced Pothole Detection System</div>', unsafe_allow_html=True)
    
    # About section
    with st.expander("ℹ️ About this project"):
        st.markdown("""
        **RoadGuardian AI** is an advanced computer vision system that detects potholes in images and videos using YOLOv5.
        
        This project was developed for the Hackathon to demonstrate how AI can help with infrastructure maintenance.
        
        **Features:**
        - Real-time pothole detection in images and videos
        - Advanced analytics and visualization
        - Severity classification of detected potholes
        - Road condition assessment and recommendations
        
        **Technical Stack:**
        - YOLOv5 for object detection
        - PyTorch for deep learning
        - OpenCV for image/video processing
        - Streamlit for the web interface
        """)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["yolov5/runs/train/exp/weights/best.pt"],
        help="Choose the trained YOLOv5 model"
    )
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence threshold for pothole detection"
    )
    
    # Load model
    try:
        with st.spinner("Loading AI model..."):
            model = load_model(model_path, conf_threshold)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.stop()
    
    # Main content area
    tabs = st.tabs(["📷 Image Analysis", "🎥 Video Analysis"])
    
    # Image Analysis Tab
    with tabs[0]:
        st.markdown('<div class="sub-header">Upload an image to detect potholes</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Display original image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🔍 Detect Potholes"):
                with st.spinner("AI analyzing image..."):
                    # Process image
                    model.conf = conf_threshold
                    result_img, detections, inference_time = process_image(model, image, conf_threshold)
                
                # Show results
                st.image(result_img, caption="Detection Results", use_column_width=True)
                
                # Display detection statistics
                st.markdown(f"""
                <div class="success-box">
                    ✅ Analysis complete! Found {len(detections)} potholes in {inference_time:.3f} seconds.
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed information for each detection
                if len(detections) > 0:
                    st.subheader("📋 Detection Details")
                    
                    # Create a table for detections
                    det_data = []
                    for i, detection in enumerate(detections):
                        x1, y1, x2, y2, conf, cls = detection
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        img_area = image.width * image.height
                        
                        # Determine severity based on size
                        severity = "Low"
                        if area / img_area > 0.05:
                            severity = "High"
                        elif area / img_area > 0.01:
                            severity = "Medium"
                        
                        det_data.append({
                            "ID": i+1,
                            "Confidence": f"{conf:.2f}",
                            "Size (px)": f"{int(width)}×{int(height)}",
                            "Severity": severity
                        })
                    
                    # Convert to DataFrame and display
                    df = pd.DataFrame(det_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Enhanced Road Condition Assessment
                    st.subheader("🔍 Detailed Road Analysis")
                    
                    # Calculate statistics for a more nuanced assessment
                    total_area = sum([(x2-x1)*(y2-y1) for x1,y1,x2,y2,_,_ in detections])
                    img_area = image.width * image.height
                    coverage_percent = (total_area / img_area) * 100
                    max_confidence = max([conf for _,_,_,_,conf,_ in detections])
                    
                    # Detailed size classification
                    size_distribution = {
                        "Small": sum(1 for x1,y1,x2,y2,_,_ in detections if (x2-x1)*(y2-y1)/img_area < 0.01),
                        "Medium": sum(1 for x1,y1,x2,y2,_,_ in detections if 0.01 <= (x2-x1)*(y2-y1)/img_area < 0.05),
                        "Large": sum(1 for x1,y1,x2,y2,_,_ in detections if (x2-x1)*(y2-y1)/img_area >= 0.05)
                    }
                    
                    # Advanced condition assessment
                    if size_distribution["Large"] > 2 or coverage_percent > 10:
                        condition = "Critical"
                        color = "#d9534f"  # Red
                        description = "This road section has severe damage with large potholes covering significant portions of the surface."
                        impact = "High risk of vehicle damage and potential safety hazard, especially during poor weather conditions."
                        recommendation = "Immediate repair recommended. Area should be marked with hazard signs until repairs are completed."
                        priority = "High - Schedule repair within 24-48 hours"
                        
                    elif size_distribution["Large"] > 0 or size_distribution["Medium"] > 3 or coverage_percent > 5:
                        condition = "Poor"
                        color = "#f0ad4e"  # Orange
                        description = "Significant road deterioration with multiple medium to large potholes."
                        impact = "Moderate risk of vehicle damage and reduced driving comfort. May require drivers to slow down or navigate around defects."
                        recommendation = "Prompt repair needed to prevent further degradation. Consider temporary patching if permanent repair is delayed."
                        priority = "Medium - Schedule repair within 1 week"
                        
                    elif size_distribution["Medium"] > 0 or size_distribution["Small"] > 5 or coverage_percent > 2:
                        condition = "Fair"
                        color = "#5bc0de"  # Blue
                        description = "Early signs of road deterioration with mostly small to medium potholes."
                        impact = "Minor impact on driving experience, but potential for rapid worsening if left untreated."
                        recommendation = "Plan for maintenance in the near future. Monitor for expansion of existing potholes."
                        priority = "Low - Include in regular maintenance schedule"
                        
                    else:
                        condition = "Good"
                        color = "#5cb85c"  # Green
                        description = "Minimal road damage with only a few small potholes detected."
                        impact = "Negligible impact on driving experience and vehicle condition."
                        recommendation = "Routine maintenance should be sufficient. Monitor during seasonal inspections."
                        priority = "Very Low - No immediate action required"
                    
                    # Display detailed assessment card
                    
                    html_content=textwrap.dedent(f"""
                    <div style="background-color: {color}20; border-left: 5px solid {color}; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <h3 style="color: {color}; margin-top: 0;">Road Condition: {condition}
                        Analysis Summary:
                        • Total potholes detected: {len(detections)}
                        • Size distribution: {size_distribution["Small"]} small, {size_distribution["Medium"]} medium, {size_distribution["Large"]} large<br>
                        • Road area affected: {coverage_percent:.2f}% of visible surface
                        • Detection confidence: {max_confidence:.2f} (highest)
                        
                        Description:{description}
                        
                        Potential Impact:{impact}
                        
                        Recommended Action:{recommendation}
                        
                        Maintenance Priority:{priority}
                    </div>
                    """)
                    st.markdown(html_content, unsafe_allow_html=True)
                    
                    # Visualization: Pothole distribution map
                    st.subheader("📊 Pothole Distribution Map")
                    
                    # Create a heatmap-style visualization of pothole locations
                    fig, ax = plt.subplots(figsize=(10, 6))
                    img_np = np.array(image)
                    ax.imshow(img_np)
                    
                    # Draw pothole locations with size-based markers
                    for i, (x1, y1, x2, y2, conf, _) in enumerate(detections):
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        area = (x2 - x1) * (y2 - y1)
                        area_percent = area / img_area
                        
                        # Determine marker size and color based on pothole size
                        if area_percent >= 0.05:
                            color = 'red'
                            size = 300
                            severity = "High"
                        elif area_percent >= 0.01:
                            color = 'orange'
                            size = 200
                            severity = "Medium"
                        else:
                            color = 'yellow'
                            size = 100
                            severity = "Low"
                            
                        ax.scatter(cx, cy, s=size, color=color, alpha=0.6, edgecolors='white')
                        ax.text(cx, cy, f"{i+1}", color='white', fontweight='bold', 
                                ha='center', va='center', fontsize=9)
                    
                    ax.set_title('Pothole Distribution Map')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Potential repair cost estimate
                    st.subheader("Estimated Repair Information in rupees")
                    
                    # Calculate rough repair cost based on pothole sizes
                    small_repair_cost = size_distribution["Small"] * 816  # $50 per small pothole
                    medium_repair_cost = size_distribution["Medium"] * 1200  # $150 per medium pothole
                    large_repair_cost = size_distribution["Large"] * 2000  # $350 per large pothole
                    total_cost = small_repair_cost + medium_repair_cost + large_repair_cost
                    
                    # Estimate repair time
                    if condition == "Critical":
                        estimated_time = "4-8 hours"
                        crew_size = "3-4 workers"
                    elif condition == "Poor":
                        estimated_time = "2-4 hours"
                        crew_size = "2-3 workers"
                    elif condition == "Fair":
                        estimated_time = "1-2 hours"
                        crew_size = "1-2 workers"
                    else:
                        estimated_time = "Under 1 hour"
                        crew_size = "1 worker"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Estimated Repair Cost", f"₹{total_cost}")
                    with col2:
                        st.metric("Estimated Repair Time", estimated_time)
                    with col3:
                        st.metric("Recommended Crew Size", crew_size)
                    
                    # Materials estimate
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                        <h4 style="margin-top: 0;">Estimated Materials Required:</h4>
                        <ul>
                            <li><strong>Asphalt Mix:</strong> {total_area/1000:.1f} cubic feet</li>
                            <li><strong>Asphalt Sealant:</strong> {coverage_percent*0.2:.1f} gallons</li>
                            <li><strong>Equipment:</strong> {'Heavy machinery (bobcat/roller)' if condition in ['Critical', 'Poor'] else 'Hand tools and compactor'}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.success("No potholes detected in this image. The road appears to be in good condition.")
    
   # Replace your existing Video Analysis Tab section with this updated version:
    with tabs[1]:
        st.markdown('<div class="sub-header">Upload a video to analyze road conditions</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        
        # Add processing options for performance
        with st.expander("Processing Options"):
            sample_rate = st.slider("Frame Sampling Rate", 
                                min_value=1, max_value=10, value=2, 
                                help="Process 1 out of every N frames. Higher values = faster processing but less detailed analysis.")
            resolution_scale = st.slider("Resolution Scale", 
                                    min_value=0.25, max_value=1.0, value=0.5, step=0.25,
                                    help="Scale the video resolution. Lower values = faster processing but may reduce accuracy.")
        
        if uploaded_file is not None:
            # Save uploaded video to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
            
            # Display original video
            st.video(video_path)
            
            # Process button
            if st.button("🎬 Analyze Video"):
                st.markdown("### 🔄 Processing Video")
                st.warning("This may take a while depending on video length...")
                
                # Create progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Output path
                output_path = os.path.splitext(video_path)[0] + "_detected.mp4"
                
                try:
                    # Open video capture
                    cap = cv2.VideoCapture(video_path)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Apply resolution scaling for faster processing
                    new_width = int(width * resolution_scale)
                    new_height = int(height * resolution_scale)
                    
                    # Set up video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps/sample_rate, (width, height))
                    
                    # Variables to track statistics
                    frame_count = 0
                    processed_count = 0
                    pothole_counts = []
                    processing_times = []
                    total_potholes = 0
                    frames_with_potholes = 0
                    severity_levels = {
                        "Low": 0,     # Small potholes
                        "Medium": 0,  # Medium potholes
                        "High": 0     # Large potholes
                    }
                    
                    # Check first frame to verify if it's a road
                    ret, first_frame = cap.read()
                    if not ret:
                        st.error("Failed to read video file")
                        raise ValueError("Cannot read video file")
                    
                   
                    # Reset video capture to start
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    # Process the video
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process only every N frames based on sample_rate
                        if frame_count % sample_rate == 0:
                            processed_count += 1
                            
                            # Resize frame for faster processing if scaling is applied
                            if resolution_scale < 1.0:
                                processed_frame = cv2.resize(frame, (new_width, new_height))
                            else:
                                processed_frame = frame
                            
                            # Process frame
                            start_time = time.time()
                            results = model(processed_frame)
                            processing_time = time.time() - start_time
                            processing_times.append(processing_time)
                            
                            # Get detections
                            detections = results.xyxy[0].cpu().numpy()
                            num_potholes = len(detections)
                            pothole_counts.append(num_potholes)
                            total_potholes += num_potholes
                            
                            if num_potholes > 0:
                                frames_with_potholes += 1
                                
                                # Classify severity based on bounding box size
                                for det in detections:
                                    x1, y1, x2, y2, conf, cls = det
                                    area = (x2 - x1) * (y2 - y1)
                                    area_percentage = area / (new_width * new_height)
                                    
                                    if area_percentage < 0.01:
                                        severity_levels["Low"] += 1
                                    elif area_percentage < 0.05:
                                        severity_levels["Medium"] += 1
                                    else:
                                        severity_levels["High"] += 1
                            
                            # Render results on original sized frame
                            if resolution_scale < 1.0:
                                # Scale detections back to original size
                                scaled_detections = []
                                scale_x = width / new_width
                                scale_y = height / new_height
                                
                                for det in detections:
                                    x1, y1, x2, y2, conf, cls = det
                                    scaled_det = [
                                        x1 * scale_x, y1 * scale_y, 
                                        x2 * scale_x, y2 * scale_y,
                                        conf, cls
                                    ]
                                    scaled_detections.append(scaled_det)
                                
                                # Draw on original frame
                                annotated_frame = frame.copy()
                                for det in scaled_detections:
                                    x1, y1, x2, y2, conf, cls = det
                                    # Draw rectangle
                                    cv2.rectangle(annotated_frame, 
                                                (int(x1), int(y1)), 
                                                (int(x2), int(y2)), 
                                                (0, 255, 0), 2)
                                    # Add label
                                    label = f"Pothole: {conf:.2f}"
                                    cv2.putText(annotated_frame, label, 
                                            (int(x1), int(y1) - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                # Use rendered frame from model
                                annotated_frame = results.render()[0]
                            
                            # Add text with detection stats
                            cv2.putText(annotated_frame, f"Potholes: {num_potholes}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"FPS: {1/max(processing_time, 0.001):.1f}", (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Write to output video
                            writer.write(annotated_frame)
                        
                        # Update progress
                        frame_count += 1
                        progress = min(frame_count / total_frames, 1.0)
                        status_text.text(f"Processing frame {frame_count}/{total_frames} | Potholes detected: {total_potholes}")
                        progress_bar.progress(progress)
                    
                    # Release resources
                    cap.release()
                    writer.release()
                    
                    # Prepare statistics
                    stats = {
                        'total_frames': frame_count,
                        'processed_frames': processed_count,
                        'frames_with_potholes': frames_with_potholes,
                        'pothole_frame_percentage': (frames_with_potholes / max(processed_count, 1)) * 100,
                        'total_potholes': total_potholes,
                        'avg_potholes_per_frame': total_potholes / max(processed_count, 1),
                        'avg_processing_time': sum(processing_times) / max(len(processing_times), 1),
                        'avg_fps': 1 / max(sum(processing_times) / max(len(processing_times), 1), 0.001),
                        'severity_levels': severity_levels,
                        'pothole_counts': pothole_counts
                    }
                    
                    st.success("✅ Video processing complete!")
                    
                    # Display processed video
                    st.subheader("🎦 Processed Video")
                    st.video(output_path)
                    
                    # Display analytics
                    display_video_analytics(stats)
                    
                    # Enhanced Road Condition Assessment for video
                    st.subheader("🔍 Detailed Road Analysis")
                    
                    # Detailed size classification from stats
                    size_distribution = {
                        "Small": severity_levels["Low"],
                        "Medium": severity_levels["Medium"],
                        "Large": severity_levels["High"]
                    }
                    
                    # Estimate affected area
                    total_road_frames = processed_count
                    avg_potholes_per_frame = stats['avg_potholes_per_frame']%frames_with_potholes
                    road_condition_percent = min(100, (frames_with_potholes / max(total_road_frames, 1)) * 100)
                    
                    # Advanced condition assessment
                    if size_distribution["Large"] > 10 or stats['total_potholes'] > 50:
                        condition = "Critical"
                        color = "#d9534f"  # Red
                        description = "This road section has severe damage with large potholes appearing frequently throughout the footage."
                        impact = "High risk of vehicle damage and potential safety hazard, especially during poor weather conditions."
                        recommendation = "Immediate repair recommended. Area should be marked with hazard signs until repairs are completed."
                        priority = "High - Schedule repair within 24-48 hours"
                        
                    elif size_distribution["Large"] > 5 or size_distribution["Medium"] > 20 or stats['total_potholes'] > 30:
                        condition = "Poor"
                        color = "#f0ad4e"  # Orange
                        description = "Significant road deterioration with multiple medium to large potholes throughout the video."
                        impact = "Moderate risk of vehicle damage and reduced driving comfort. May require drivers to slow down or navigate around defects."
                        recommendation = "Prompt repair needed to prevent further degradation. Consider temporary patching if permanent repair is delayed."
                        priority = "Medium - Schedule repair within 1 week"
                        
                    elif size_distribution["Medium"] > 10 or size_distribution["Small"] > 20 or stats['total_potholes'] > 15:
                        condition = "Fair"
                        color = "#5bc0de"  # Blue
                        description = "Early signs of road deterioration with mostly small to medium potholes appearing periodically."
                        impact = "Minor impact on driving experience, but potential for rapid worsening if left untreated."
                        recommendation = "Plan for maintenance in the near future. Monitor for expansion of existing potholes."
                        priority = "Low - Include in regular maintenance schedule"
                        
                    else:
                        condition = "Good"
                        color = "#5cb85c"  # Green
                        description = "Minimal road damage with only a few small potholes detected throughout the video."
                        impact = "Negligible impact on driving experience and vehicle condition."
                        recommendation = "Routine maintenance should be sufficient. Monitor during seasonal inspections."
                        priority = "Very Low - No immediate action required"
                    
                    # Display detailed assessment card
                    st.markdown(f"""
                    <div style="background-color: {color}20; border-left: 5px solid {color}; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <h3 style="color: {color}; margin-top: 0;">Road Condition: {condition}</h3>
                        Analysis Summary:
                        • Total potholes detected: {stats['total_potholes']}<br>
                        • Size distribution: {size_distribution["Small"]} small, {size_distribution["Medium"]} medium, {size_distribution["Large"]} large<br>
                        • Road condition issues in: {road_condition_percent:.1f}% of the video
                        • Average potholes per analyzed frame: {avg_potholes_per_frame:.2f}
                        
                        Description:{description}
                        
                        Potential Impact:{impact}
                        
                        Recommended Action:{recommendation}
                        
                        Maintenance Priority:{priority}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Potential repair cost estimate
                    st.subheader("Estimated Repair Information in rupees")
                    
                    # Calculate rough repair cost based on pothole sizes
                    small_repair_cost = size_distribution["Small"] * 816
                    medium_repair_cost = size_distribution["Medium"] * 1200
                    large_repair_cost = size_distribution["Large"] * 2000
                    total_cost = small_repair_cost + medium_repair_cost + large_repair_cost
                    
                    # Estimate repair time
                    if condition == "Critical":
                        estimated_time = "1-2 days"
                        crew_size = "4-6 workers"
                    elif condition == "Poor":
                        estimated_time = "4-8 hours"
                        crew_size = "3-4 workers"
                    elif condition == "Fair":
                        estimated_time = "2-4 hours"
                        crew_size = "2-3 workers"
                    else:
                        estimated_time = "Under 2 hours"
                        crew_size = "1-2 workers"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Estimated Repair Cost", f"₹{total_cost}")
                    with col2:
                        st.metric("Estimated Repair Time", estimated_time)
                    with col3:
                        st.metric("Recommended Crew Size", crew_size)
                    
                    # Materials estimate
                    road_length_estimate = "unknown"
                    if "road length" in uploaded_file.name.lower():
                        # Try to extract road length from filename if provided
                        try:
                            import re
                            match = re.search(r'(\d+)m', uploaded_file.name.lower())
                            if match:
                                road_length_estimate = f"{match.group(1)} meters"
                        except:
                            pass
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                        <h4 style="margin-top: 0;">Estimated Materials Required:</h4>
                        <ul>
                            <li><strong>Asphalt Mix:</strong> {total_potholes * 0.5:.1f} cubic feet</li>
                            <li><strong>Asphalt Sealant:</strong> {total_potholes * 0.2:.1f} gallons</li>
                            <li><strong>Equipment:</strong> {'Heavy machinery (bobcat/roller)' if condition in ['Critical', 'Poor'] else 'Hand tools and compactor'}</li>
                            <li><strong>Estimated Road Section:</strong> {road_length_estimate}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except ValueError as ve:
                    if "Not a road video" in str(ve):
                        # Error already displayed
                        pass
                    else:
                        st.error(f"Error processing video: {ve}")
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                
                finally:
                    try:
                        os.unlink(video_path)
                    except:
                        pass


if __name__ == "__main__":
    main()
