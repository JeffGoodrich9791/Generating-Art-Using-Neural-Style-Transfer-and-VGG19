# Generating-Art-Using-Neural-Style-Transfer-and-VGG19
## Deep Learning & Art: Neural Style Transfer 

<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/First ConvNet Artwork.png" />

### Summary

The goal of this project was to utilize the Nerual Style Transfer (NST) algorithms to generate a unique artwork from two input images. The two images for input were the "content component" image and the "style component" image. The content component images is what the convolutional neural network (CNN) used as structure for the image genteration. The style component image was then combined with the content image in several different layers of the CNN. Each layer of the CNN acts as a filter to extract certain features or attributes of each feature such as texture, corners, small shapes, or repeating color patterns. The lower level dimensionality features of the input images are filtered and learned in the first few layers of the CNN. These layers would be responsible for noticing color patterns and small repeating patterns. The higher level dimensionality features responsible for actual content creation such as buildings, and objects are filtered and learned in higher level layers of the CNN. Of course some of the layers responsible for both content and style overlap in the CNN, and this is where the NST actually takes place. 

The original arrangement of the photograph content is preserved, while the colors and minor structures that compose the style  become entangled. Therefore the generated image has a new unique style even though it shows the same content as the original photograph. 


### Model

Template code is provided in the `Generating Art Using Neural Style Transfer and VGG19.ipynb` notebook file. The VGG19 model consists of 19 layers and was constructed using Python 3 and Tensorflow in an iPyton Notebook. The input images were of shape 400px X 300px X 3 (rgb). The input images are used are displayed below:
<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/louvre.jpg" />
<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/sandstone.jpg" />

The VGG19 network used has already been pre-trained on a large number of images from the ImageNet database therefore it has learned the weights for low-level features and high-level features. The NST algorithm is developed by computing the cost of the content image, cost of the style image, and finally the cost of the generated image. The cost function of the style implements the Gram Matrix in the computation, also called the style matrix.  

The cost of the content is computed using the content image (C) and generated image (G):

<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/Jcost_content.png" />

The Gram Matrix, is computed: 

<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/Gram_matrix.png" />

The cost of the style is computed using the style image (S) and generated image (G) and the mean squared error of the Gram Matrix:

<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/Jcost_style.png" />



### Run

The output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion. The following cell does executes this. 

> yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

The yolo_outputs produced all of the predicted boxes of yolo_model in the correct format. Now the the ouptuts need to be filtered using this command. 

> scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

The prediction for the new image is then run through the following code: 

>  Preprocess your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input: image_data,    K.learning_phase():0})
    ### END CODE HERE ###

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes


### Results

> out_scores, out_boxes, out_classes = predict(sess, "test.jpg")

<img src= "https://github.com/JeffGoodrich9791/YOLOv2_Autonomous_Vehicle_Image_Detection/blob/master/Bounding_Box_Output.png" /> 

