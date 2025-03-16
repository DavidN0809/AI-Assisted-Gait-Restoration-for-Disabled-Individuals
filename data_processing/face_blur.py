
    def blur_faces_in_images(self):
        """Blur faces in saved images using a pre-trained Caffe model."""
        # Load the pre-trained face detection model
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        # np.load(picture_info["image_filename"], allow_pickle=True)[0]['img_data']
        # Iterate through all the saved images in the action folder
        for root, dirs, files in os.walk(self.collection_data_handler.action_folder):
            for file in files:
                if file.endswith(".npy"):
                    image_path = os.path.join(root, file)
                    print(f"Processing image: {image_path}")
                    
                    # Load the image
                    # image = cv2.imread(image_path)
                    img_info = np.load(image_path, allow_pickle=True)
                    image = img_info[0]['img_data']
                    (h, w) = image.shape[:2]
                    
                    # Prepare the image for face detection
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()
                    
                    # Iterate over the detections and blur the faces
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        
                        # Filter out weak detections
                        if confidence > 0.5:
                            # Get the coordinates of the bounding box for the face
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            # Ensure the bounding box is within the dimensions of the image
                            startX, startY = max(0, startX), max(0, startY)
                            endX, endY = min(w, endX), min(h, endY)
                            
                            # Extract the face region
                            face = image[startY:endY, startX:endX]
                            
                            # Apply a Gaussian blur to the face region
                            face = cv2.GaussianBlur(face, (99, 99), 30)
                            
                            # Put the blurred face back into the image
                            image[startY:endY, startX:endX] = face
                    
                    # Save the image (overwrite og image)
                    # cv2.imwrite(image_path[:-3] + "jpg", image)
                    img_info[0]['img_data'] = image
                    np.save(image_path, img_info)