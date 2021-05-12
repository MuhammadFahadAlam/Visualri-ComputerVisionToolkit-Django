import os
import cv2
import base64
import numpy as np
from io import BytesIO
from django.conf import settings
# pip install opencv-contrib-python

## commenting right now

#______import bleedfacedetector as fd


# pip install dlib # had to use conda for dlib ...was unable to download it using pip
# dlib is prerequisite for bleedfacedetector
# pip install bleedfacedetector
# pip install opencv-contrib-python

def color_filter(image,action):
    
    return cv2.applyColorMap(image, int(action))[:,:,::-1]



model_name= ['rain','pink','triangle','gold_black','flame','Fire_Style','landscape','feathers', 'candy', 'composition_vii', 'udnie', 'the_wave', 'the_scream', 'mosaic', 'la_muse', 'starry_night']

def style_filter(image,action ):
    net = init_style_transfer(action)
    styled = style_tranfer(image,net)
    return styled[:,:,::-1]

base_path = str(settings.BASE_DIR) + '/visualri/static/visualri/style-model/'

def init_style_transfer(style="candy"):
    if style in model_name:
        print('True' )
        model =base_path+str(style) +".t7"
        print(model)
        net = cv2.dnn.readNetFromTorch(model)
    return net

def style_tranfer(image,net):
    R,G,B = 103.939, 116.779, 123.680
    blob = cv2.dnn.blobFromImage(image, 1.0, (image.shape[1], image.shape[0]),(R,G,B), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    final_output = output.reshape((3, output.shape[2], output.shape[3])).copy()
    final_output[0] += R
    final_output[1] += G
    final_output[2] += B
    final_output = final_output.transpose(1, 2, 0)
    outmid = np.clip(final_output, 0, 255)
    styled= outmid.astype('uint8')
    return styled
    


def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')

def get_filtered_image(image, action):
    if action == 'THRESHOLDING':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,5)
        filtered = mean_thresh

    elif action == 'BGR':        
        filtered = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif action == 'GRAYSCALE':
            filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif action == 'HSV':
        filtered = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    elif action == 'STYLIZATION':
        filtered = cv2.stylization(image, sigma_s=50, sigma_r=0.57)

    elif action == 'DETAIL ENHANCE':
        filtered = cv2.detailEnhance(image, sigma_s=30, sigma_r=0.15)

    elif action == 'COLOR SKETCH':
        dst_gray, filtered = cv2.pencilSketch(image, sigma_s=50, sigma_r=0.09, shade_factor=0.05)

    elif action == 'EDGE PRESERVING FILTER':
        filtered = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)

    elif action == 'MEDIAN BLUR':
        filtered = cv2.medianBlur(image, 5)

    elif action == 'BLURRED':
        filtered = bi_blur(image)

    elif action == 'POLARIZE':
        filtered = polarize(image)

    elif action == 'BRIGHTNESS':
        filtered = brightness(image)

    elif action == 'CONTRAST':
        filtered = contrast(image)

    elif action == 'INPAINT':
        filtered = inpaint(image)

    elif action == 'COLORIZED':
        net = init_colorized()
        filtered = colorization(image,net)

    elif action == 'SUPERRES':
        net = init_superres()
        filtered = super_res(image,net)

    return filtered


def inpaint(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # As simple as that you have a mask
    ret, mask = cv2.threshold(gray_img, 250, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask,None,iterations=2)
    result_ns = cv2.inpaint(image,mask,3,cv2.INPAINT_NS)
    return result_ns

def bi_blur(image):
    SigmaColor = 60
    SigmaSpace = 60
    bi_blur = cv2.bilateralFilter(image, 9, SigmaColor, SigmaSpace)
    return bi_blur

def polarize(image):
    pixel_values = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255])
    # make the dark pixels darker and bright pixels brighter
    new_values = np.array([0, 10, 25, 40, 65, 125, 180, 210, 235, 245, 255])
    # create the lookUP Table
    pixel_range = np.arange(0, 256)
    LUT = np.interp(pixel_range, pixel_values, new_values)
    # converting to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # modify the value channel
    hsv[:, :, 2] = cv2.LUT(hsv[:, :, 2], LUT)
    # convert back to BGR
    filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return filtered

def brightness(image):
    lookUpTable = np.zeros((1, 256), np.uint8)
    # setting the mapping value of gamma
    gamma = 0.4
    # mapping each value from 0 to 255 to its required power and then clipping it
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    final = cv2.LUT(image, lookUpTable)
    final=cv2.flip(final,1)
    final=cv2.flip(final,1)
    return final

def contrast(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # performing histogram equalization only on the value Channel
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    # Converting back to BGR format
    filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return filtered



def init_colorized(weights_name='colorization_deploy_v2.prototxt',architecture_name='colorization_release_v2.caffemodel'):
    base_path = str(settings.BASE_DIR)+'/Media/M4/Model'
    protofile = os.path.join(base_path, weights_name)
    model = os.path.join(base_path, architecture_name)
    # Load the points
    points_path = os.path.join(base_path, 'pts_in_hull.npy')
    points = np.load(points_path)
    # Format the points into required shape.
    points = points.transpose()
    points = points.reshape(2, 313, 1, 1)
    # Convert to float data type and into a list format.
    points_float32 = [points.astype("float32")]
    # Intialize the model
    net = cv2.dnn.readNetFromCaffe(protofile, model)
    # Here we are fetching the required layer IDs
    cls_8 = net.getLayerId("class8_ab")
    conv_8 = net.getLayerId("conv8_313_rh")
    # Here we are loading cluster ab points into a specific parts of the net.
    net.getLayer(cls_8).blobs = points_float32
    # This is a scaling layer used for normalization.
    net.getLayer(conv_8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    # If specified use either cuda based Nvidia gpu or opencl based Intel gpu.
    return net

def colorization(image,net):
    # Changing the data type to float32 and divide it 255 so the range should be b/w 0 - 1
    scaled_image = image.astype("float32") / 255.0
    # Here we are converting the BGR image into the lab color space
    lab_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2LAB)
    # Colorization network requires the image to be in 224x224
    lab_image_resized = cv2.resize(lab_image, (224, 224))
    # Fetching the L channel from resized lab color space
    l_channel = lab_image_resized[:, :, 0]
    # Mean subtraction as this was done by the authors as a preprocessing step.
    l_channel -= 50
    # Make a blob from the l channel
    blob = cv2.dnn.blobFromImage(l_channel)
    # set the L channel as input.
    net.setInput(blob)
    # Predicted a,b channel
    ab_channel = net.forward()
    # Reshape the predicted a,b channels
    ab_channel_resized = ab_channel[0, :, :, :].transpose((1, 2, 0))
    # Here we are resizing the predicted a,b channels with simillar dimension of our original image
    a_b = cv2.resize(ab_channel_resized, (image.shape[1], image.shape[0]))
    # Here we are fetching L chanel from original image
    l_channel = lab_image[:, :, 0]
    # Concatenate both L channel from  original image and ab channel which are predicted
    colored = np.concatenate((l_channel[:, :, np.newaxis], a_b), axis=2)
    # Convert Lab image to BGR.
    colored = cv2.cvtColor(colored, cv2.COLOR_LAB2BGR)
    # clip the values below 0 and above 1.
    colored = np.clip(colored, 0, 1)
    # Multiply by 255 so the range is b/w 0-255 and convert to uint8.
    colored = (255 * colored).astype("uint8")
    return colored[:,:,::-1]


def init_superres(model="super_resolution.onnx"):
    base_path = str(settings.BASE_DIR)+'/Media/M4/Model'

    # Defining model path
    model = os.path.join(base_path, model)

    # Initialize the DNN module
    net = cv2.dnn.readNetFromONNX(model)

    # If specified use either cuda based Nvidia gpu or opencl based Intel gpu.
    return net


def super_res(image,net):
    # Creating Copy of Image
    img_copy = image.copy()
    # Resize the image into Required Size
    img_copy = cv2.resize(img_copy, (224, 224), cv2.INTER_CUBIC)
    # Convert image into YcbCr
    image_YCbCr = cv2.cvtColor(img_copy, cv2.COLOR_BGR2YCrCb)
    # Split Y,Cb, and Cr channel
    image_Y, image_Cb, image_Cr = cv2.split(image_YCbCr)
    # Convert Y channel into a numpy arrary
    img_ndarray = np.asarray(image_Y)
    # Reshape the image to (1,1,224,224)
    image_expanded = img_ndarray.reshape(1, 1, 224, 224)
    # Convert to float32 and as a normalization step divide the image by 255.0
    blob = image_expanded.astype(np.float32) / 255.0
    # Passing the blob as input through the network
    net.setInput(blob)
    # Forward pass
    Output = net.forward()
    # Reshape the output and get rid of those extra dimensions
    reshaped_output = Output.reshape(672, 672)
    # Get the image back to the range 0-255 from 0-1
    reshaped_output = reshaped_output * 255
    # Clip the values so the output is it between 0-255
    Final_Output = np.clip(reshaped_output, 0, 255)
    # Resize the Cb and Cr channel with output dimension
    resize_Cb = cv2.resize(image_Cb, (672, 672), cv2.INTER_CUBIC)
    resized_Cr = cv2.resize(image_Cr, (672, 672), cv2.INTER_CUBIC)
    # Merge 3 channel together
    Final_Img = cv2.merge((Final_Output.astype('uint8'), resize_Cb, resized_Cr))
    # Convert back into BGR channel
    Final_Img = cv2.cvtColor(Final_Img, cv2.COLOR_YCR_CB2RGB)
    # This is how the image would look with Bicubic interpolation.
    Final_Img = cv2.resize(Final_Img, (image.shape[1],image.shape[0]), cv2.INTER_CUBIC)
    # Stack two image together inorder to see the comparision b/w them
    return Final_Img[:,:,::-1]


'''
def emotion_filter(image):
    net= init_emotion()
    filtered = emotion(image,net)
    return filtered



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
    img_copy = image.copy()
    # Detect face in image
    faces = fd.ssd_detect(img_copy, conf=0.30)
    if len(faces) == 0:
        return img_copy
    
    padding = 12
    # Iterate for each detected faces & apply the above process for each face.
    for x, y, w, h in faces:
        # Padd the face
        try:
            face = img_copy[y - padding:y + h + padding, x - padding:x + w + padding]
            face = cv2.resize(face[:, :, 1], (64, 64))
            imager = face.reshape(1, 1, 64, 64)
            net.setInput(imager)
            # Perfrom the forward pass.
            output = net.forward()

            # Get the predicted age group.
            predicted_emotions = emotions[output.argmax()]

            # Draw the bounding box around the face and put the age group.
            cv2.putText(img_copy, '{}'.format(predicted_emotions), (x, y + h+5 + (1 * 20)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        except: 
            continue
    return img_copy


def gender_filter(image):
    net = init_detectgender()
    return detectgender(image,net)


def init_detectgender(weights_name="gender_net.caffemodel",proto_name="gender_deploy.prototxt"):
     
    # Defining base path
    base_path = str(settings.BASE_DIR)+'/Media/M4/Model'
   # Defining model paths
    proto_file = os.path.join(base_path, proto_name)
    weights = os.path.join(base_path, weights_name) 
    # Initialize the DNN module
    net = cv2.dnn.readNet(weights,proto_file)
    return net
    


def detectgender(image,net,face_conf=0.2):
     # Define Gender List
    Genders= ['Male', 'Female']
    img_copy = image.copy()
    # Use SSD detector with 20% confidence threshold.
    faces = fd.ssd_detect(img_copy, conf=face_conf)
    # Lets take coordinates of the first face in the image. 
    if len(faces) == 0:
        return img_copy    
    x,y,w,h = faces[0]
    # Define padding for face roi
    padding = 3
    # extract the Face from image with padding.
    face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding] 
    # Prepare the frame to be fed to the network
    blob  = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    net.setInput(blob)
    output = net.forward()
    padding = 3 
    for x,y,w,h in faces:
        try:
            face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding]
            print(face.shape)
            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            net.setInput(blob)
            output = net.forward()
            predicted_gender = Genders[output[0].argmax()]
            cv2.putText(img_copy,'{}'.format(predicted_gender),(x,y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2, cv2.LINE_AA)
            cv2.rectangle(img_copy,(x-10,y-25),(x+w,y+h),(0,0,255),2)
        except:
            continue
    return img_copy
'''