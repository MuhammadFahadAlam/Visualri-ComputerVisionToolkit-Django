#import bleedfacedetector as fd
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
face_detection_videocam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
face_detection_webcam = cv2.CascadeClassifier(os.path.join(
			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
		

	def __del__(self):
		self.video.release()

	def get_frame(self,model_instance,net):
		self.action = model_instance.action
		success, image = self.video.read()
		self.frame = image
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream.

		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
#		print(self.action)
		if self.action == "GENDER":
			image = detectgender(image,net,face_conf=model_instance.confidence)
		
		elif self.action == "EMOTION":
			image = emotion(image,net)
			
		elif self.action =="FACE":
			image = detect_face(image,face_conf=model_instance.confidence)
		
		else:
			image = cv2.flip(image,1)

		ret, jpeg = cv2.imencode('.jpg', image)
		return jpeg.tobytes()
	

class IPWebCam(object):
	def __init__(self,ip):
		self.url = "http://"+ip+"/shot.jpg"

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self,model_instance,net):
		self.action = model_instance.action

		imgResp = urllib.request.urlopen(self.url)
		imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
		img= cv2.imdecode(imgNp,-1)
		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
		# so we must encode it into JPEG in order to correctly display the
		# video stream
		#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=model_instance.scale_factor, minNeighbors=model_instance.neighbours)
		
		if self.action == "GENDER":
			img = detectgender(img,net,face_conf=model_instance.confidence)
		
		elif self.action == "EMOTION":
			img = emotion(img,net)
			print("In else")
		elif self.action=="FACE":
			img = detect_face(img,face_conf=model_instance.confidence)
		else:
			img = cv2.flip(image,1)

		resize = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR)
		
		ret, jpeg = cv2.imencode('.jpg', resize)
		return jpeg.tobytes()

def detect_face(image,face_conf=0.3):
	img_copy = cv2.flip(image,1)
	#faces = fd.ssd_detect(img_copy, conf=face_conf)
    
	faces = face_detection_videocam.detectMultiScale(img_copy, scaleFactor=1.3, minNeighbors=5)

	if len(faces) == 0:
		return img_copy  
	
	# Define padding for face roi
	padding = 3
	# extract the Face from image with padding.
		
	padding = 3 
	for x,y,w,h in faces:
		try:
			cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
		except:		
			continue
	return img_copy


def init_detectgender(weights_name="gender_net.caffemodel",proto_name="gender_deploy.prototxt"):
    
    # Defining base path
    base_path = str(settings.BASE_DIR)+'/Media/M4/Model'
   # Defining model paths
    proto_file = os.path.join(base_path, proto_name)
    weights = os.path.join(base_path, weights_name) 
    # Initialize the DNN module
    net = cv2.dnn.readNet(weights,proto_file)
    
    return net


def detectgender(image,net,face_conf=0.3,faces=''):
	
	img_copy = cv2.flip(image,1)
	Genders= ['Male', 'Female']
    # Define Gender List
	#faces = fd.ssd_detect(img_copy, conf=face_conf)
    
	faces = face_detection_videocam.detectMultiScale(img_copy, scaleFactor=1.3, minNeighbors=5)

	if len(faces) == 0:
		return img_copy  
	x,y,w,h = faces[0]
	# Define padding for face roi
	padding = 3
	# extract the Face from image with padding.
	try:
		face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding] 
		# Prepare the frame to be fed to the network
		blob  = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
		net.setInput(blob)
		output = net.forward()
	except:
		return img_copy
	padding = 3 
	for x,y,w,h in faces:
		try:
			face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding]
			#print(face.shape)
			blob = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
			net.setInput(blob)
			output = net.forward()
			predicted_gender = Genders[output[0].argmax()]
			
			cv2.putText(img_copy,'{}'.format(predicted_gender),(x,y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2, cv2.LINE_AA)
			cv2.rectangle(img_copy,(x-10,y-25),(x+w,y+h),(0,0,255),2)
		except:
		
			continue
	return img_copy

def init_emotion( model="emotion-ferplus-8.onnx"):
    # Defining model path
    base_path = str(settings.BASE_DIR)+'/Media/M4/Model'
    model = os.path.join(base_path, model)
    # Initialize the DNN module
    net = cv2.dnn.readNetFromONNX(model)
    # If specified use either cuda based Nvidia gpu or opencl based Intel gpu.
    return net


def emotion(image,net):
	emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
	img_copy = cv2.flip(image,1)    # Detect face in image
	#faces = fd.ssd_detect(img_copy, conf=0.30)
	
	faces = face_detection_videocam.detectMultiScale(img_copy, scaleFactor=1.3, minNeighbors=5)
	
	if len(faces) == 0:
		return img_copy

	padding = 10
	# Iterate for each detected faces & apply the above process for each face.
	for x, y, w, h in faces:
		# Padd the face
		try:
			face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding]
			face = cv2.resize(face[:,:,1], (64, 64))
					
			imager = face.reshape(1,1,64,64)
			net.setInput(imager)
			# Perfrom the forward pass.
			output = net.forward()
			# Get the predicted age group.
			predicted_emotions = emotions[output.argmax()]
			# Draw the bounding box around the face and put the age group.
			cv2.putText(img_copy,'{}'.format(predicted_emotions),(x,y+h+(1*20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2, cv2.LINE_AA)
			cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
			# Perfrom the forward pass.
			# Draw the bounding box around the face and put the age group.
		except: 
			continue
	return img_copy

