import base64
from io import BytesIO
import cv2
import numpy as np
import os
import statistics as st
import bleedfacedetector as fd
from django.conf import settings

def color_filter(image,action):
    
    return cv2.applyColorMap(image, int(action))[:,:,::-1]



model_name= ['rain','pink','triangle','gold_black','flame','Fire_Style','landscape','feathers', 'candy', 'composition_vii', 'udnie', 'the_wave', 'the_scream', 'mosaic', 'la_muse', 'starry_night']

def style_filter(image,action ):
    net = init_style_transfer(action)
    styled = style_tranfer(image,net)
    return styled

base_path = ''

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

def blur(img,kernel):
    blurred = cv2.blur(img, (kernel, kernel))
    return blurred


def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')


# pip install opencv-contrib-python
#import bleedfacedetector as fd


# pip install dlib # had to use conda for dlib ...was unable to download it using pip
# dlib is prerequisite for bleedfacedetector
# pip install bleedfacedetector
# pip install opencv-contrib-python

def get_filtered_image(image, action):
    if action == 'NO_FILTER':
        filtered = image

    elif action == 'RGB':
        filtered = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif action == 'HSV':
        filtered = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    elif action == 'GRAYSCALE':
        filtered = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif action == 'BLURRED':
        SigmaColor = 60
        SigmaSpace = 60
        bi_blur = cv2.bilateralFilter(image, 9, SigmaColor, SigmaSpace)
        filtered = bi_blur

    elif action == 'POLARIZE':
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

    elif action == 'COOLING':
        pixel_values = np.array([0, 50, 100, 150, 200, 255])
        # map the pixel_values to a higher value for the blue channel
        blue_Channel = np.array([0, 75, 140, 180, 220, 255])
        # map the pixel_values to a higher value for the blue channel
        red_Channel = np.array([0, 25, 55, 80, 160, 255])
        pixel_range = np.arange(0, 256)
        red_channel_LUT = np.interp(pixel_range, pixel_values, red_Channel)
        blue_channel_LUT = np.interp(pixel_range, pixel_values, blue_Channel)
        # mapping blue channel to new values
        image[:, :, 0] = cv2.LUT(image[:, :, 0], blue_channel_LUT)

        # mapping red channel to new values
        image[:, :, 2] = cv2.LUT(image[:, :, 2], red_channel_LUT)

        filtered = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif action == 'WARMING':
        pixel_values = np.array([0, 50, 100, 150, 200, 255])
        # map the pixel_values to a higher value for the blue channel
        blue_Channel = np.array([0, 75, 140, 180, 220, 255])
        # map the pixel_values to a higher value for the blue channel
        red_Channel = np.array([0, 25, 55, 80, 160, 255])
        pixel_range = np.arange(0, 256)
        red_channel_LUT = np.interp(pixel_range, pixel_values, red_Channel)
        blue_channel_LUT = np.interp(pixel_range, pixel_values, blue_Channel)
        # mapping blue channel to new values
        image[:, :, 0] = cv2.LUT(image[:, :, 0], blue_channel_LUT)

        # mapping red channel to new values
        image[:, :, 2] = cv2.LUT(image[:, :, 2], red_channel_LUT)

        filtered = image

    elif action == 'BRIGHTNESS':
        lookUpTable = np.zeros((1, 256), np.uint8)
        # setting the mapping value of gamma
        gamma = 0.4

        # mapping each value from 0 to 255 to its required power and then clipping it
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        final = cv2.LUT(image, lookUpTable)

        # convert back to uint8
        filtered = final
        #filtered = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

    elif action == 'CONTRAST':
        #print(image.shape)
        #image = cv2.imread(os.path.join(), 0)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # performing histogram equalization only on the value Channel
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

        # Converting back to BGR format
        filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


    elif action == 'COLORIZED':
        net, points = init_colorized()
        colored = colorization(image,net)
        filtered = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    elif action == 'SUPERRES':
        net = init_superres()
        res = super_res(image,net)
        filtered = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    elif action == 'STYLIZATION':
        filtered = cv2.stylization(image, sigma_s=50, sigma_r=0.57)
    elif action == 'DETAIL ENHANCE':
        filtered = cv2.detailEnhance(image, sigma_s=30, sigma_r=0.15)
    elif action == 'EDGE PRESERVING FILTER':
        filtered = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
    elif action == 'COLOR SKETCH':
        dst_gray, filtered = cv2.pencilSketch(image, sigma_s=50, sigma_r=0.09, shade_factor=0.05)
    elif action == 'MEDIAN BLUR':
        filtered = cv2.medianBlur(image, 5)
    # elif action == 'EMOTION':
    #     net, emotions = init_emotion()
    #     img = emotion(image)
    #     filtered = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return filtered


def init_colorized(usegpu='None', weights_name='colorization_deploy_v2.prototxt',
                   architecture_name='colorization_release_v2.caffemodel'):
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
    if usegpu == 'cuda':
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    elif usegpu == 'opencl':
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    return net, points


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
    return colored


def init_superres(usegpu='None', model="super_resolution.onnx"):
    base_path = str(settings.BASE_DIR)+'/Media/M4/Model'

    # Defining model path
    model = os.path.join(base_path, model)

    # Initialize the DNN module
    net = cv2.dnn.readNetFromONNX(model)

    # If specified use either cuda based Nvidia gpu or opencl based Intel gpu.
    if usegpu == 'cuda':
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    elif usegpu == 'opencl':
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    return net


def super_res(image,net):
    # If the user did'nt specified the image then consider then consider choosing file or camera snapshot.

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
    Final_Img = cv2.cvtColor(Final_Img, cv2.COLOR_YCR_CB2BGR)

    # This is how the image would look with Bicubic interpolation.
    image_copy = cv2.resize(image, (672, 672), cv2.INTER_CUBIC)

    # Stack two image together inorder to see the comparision b/w them
    stacked = np.hstack((image_copy, Final_Img))

    return stacked


# def init_emotion(usegpu='None', model="emotion-ferplus-8.onnx"):
#     # Set global variables
#
#     base_path = str(settings.BASE_DIR)+'/Media/M4/Model'
#
#     # Define the emotions
#     emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
#     # Defining model path
#     model = os.path.join(base_path, model)
#
#     # Initialize the DNN module
#     net = cv2.dnn.readNetFromONNX(model)
#
#     # If specified use either cuda based Nvidia gpu or opencl based Intel gpu.
#     if usegpu == 'cuda':
#         net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#         net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#
#     elif usegpu == 'opencl':
#         net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
#     return net, emotions
#
#
# def emotion(image):
#     # If the user did'nt specified the image then consider then consider choosing file or camera snapshot.
#     img_copy = image.copy()
#
#     # Detect face in image
#     faces = fd.ssd_detect(img_copy, conf=0.2)
#
#     if len(faces) == 0:
#         return None
#
#     padding = 10
#
#     # Iterate for each detected faces & apply the above process for each face.
#     for x, y, w, h in faces:
#         # Padd the face
#         face = img_copy[y - padding:y + h + padding, x - padding:x + w + padding]
#
#         face = cv2.resize(face[:, :, 1], (64, 64))
#         imager = face.reshape(1, 1, 64, 64)
#
#         net.setInput(imager)
#
#         # Perfrom the forward pass.
#         output = net.forward()
#
#         # Get the predicted age group.
#         predicted_emotions = emotions[output.argmax()]
#
#         # Draw the bounding box around the face and put the age group.
#         cv2.putText(img_copy, '{}'.format(predicted_emotions), (x, y + h + (1 * 20)), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (255, 0, 255), 2, cv2.LINE_AA)
#         cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#     return img_copy

# def get_filtered_image(img, action):
#
#     if action == 'NO_FILTER':
#         filtered = img
#
#     elif action == 'COLORIZED':
#         filtered = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     elif action == 'GRAYSCALE':
#         filtered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     elif action == 'BLURRED':
#         SigmaColor = 60
#         SigmaSpace = 60
#         bi_blur = cv2.bilateralFilter(img, 9, SigmaColor, SigmaSpace)
#
#         filtered = bi_blur
#     elif action == 'POLARIZE':
#         pixel_values = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255])
#         # make the dark pixels darker and bright pixels brighter
#         new_values = np.array([0, 10, 25, 40, 65, 125, 180, 210, 235, 245, 255])
#         # create the lookUP Table
#         pixel_range = np.arange(0, 256)
#         LUT = np.interp(pixel_range, pixel_values, new_values)
#
#         # converting to hsv
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         # modify the value channel
#         hsv[:, :, 2] = cv2.LUT(hsv[:, :, 2], LUT)
#
#         # convert back to BGR
#         filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     elif action == 'COOLING':
#         pixel_values = np.array([0, 50, 100, 150, 200, 255])
#         # map the pixel_values to a higher value for the blue channel
#         blue_Channel = np.array([0, 75, 140, 180, 220, 255])
#         # map the pixel_values to a higher value for the blue channel
#         red_Channel = np.array([0, 25, 55, 80, 160, 255])
#         pixel_range = np.arange(0, 256)
#         red_channel_LUT = np.interp(pixel_range, pixel_values, red_Channel)
#         blue_channel_LUT = np.interp(pixel_range, pixel_values, blue_Channel)
#         # mapping blue channel to new values
#         img[:, :, 0] = cv2.LUT(img[:, :, 0], blue_channel_LUT)
#
#         # mapping red channel to new values
#         img[:, :, 2] = cv2.LUT(img[:, :, 2], red_channel_LUT)
#
#         filtered = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     elif action == 'WARMING':
#         pixel_values = np.array([0, 50, 100, 150, 200, 255])
#         # map the pixel_values to a higher value for the blue channel
#         blue_Channel = np.array([0, 75, 140, 180, 220, 255])
#         # map the pixel_values to a higher value for the blue channel
#         red_Channel = np.array([0, 25, 55, 80, 160, 255])
#         pixel_range = np.arange(0, 256)
#         red_channel_LUT = np.interp(pixel_range, pixel_values, red_Channel)
#         blue_channel_LUT = np.interp(pixel_range, pixel_values, blue_Channel)
#         # mapping blue channel to new values
#         img[:, :, 0] = cv2.LUT(img[:, :, 0], blue_channel_LUT)
#
#         # mapping red channel to new values
#         img[:, :, 2] = cv2.LUT(img[:, :, 2], red_channel_LUT)
#
#         filtered = img
#
#     elif action == 'COLORMAP':
#         filtered = cv2.applyColorMap(img, cv2.COLORMAP_PLASMA)
#         # 20 color maps will take input from user
#
#
#     elif action == 'BRIGHTNESS':
#
#         lookUpTable = np.zeros((1, 256), np.uint8)
#
#         # setting the mapping value of gamma
#         gamma = 0.4
#
#         # mapping each value from 0 to 255 to its required power and then clipping it
#         for i in range(256):
#             lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
#
#         final = cv2.LUT(img, lookUpTable)
#
#         # convert back to uint8
#
#         filtered = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
#
#     elif action == 'CONTRAST':
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#         # performing histogram equalization only on the value Channel
#         hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
#
#         # Converting back to BGR format
#         filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#     elif action == 'DominantColor':
#         hsvImage = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#         H, S, V = cv2.split(hsvImage)
#         H_array = H[S > 15].flatten()
#         hue = st.mode(H_array)
#         blank_image = np.zeros((img.shape[1], img.shape[0], 3), np.uint8)
#         blank_image = cv2.cvtColor(blank_image, cv2.COLOR_RGB2HSV)
#         blank_image[:, :, 0] = hue
#         blank_image[:, :, 1] = 0
#         blank_image[:, :, 2] = 0
#
#         filtered = blank_image
#
#     return filtered