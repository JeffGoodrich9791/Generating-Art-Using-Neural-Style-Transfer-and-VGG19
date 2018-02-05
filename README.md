# Generating-Art-Using-Neural-Style-Transfer-and-VGG19
## Deep Learning & Art: Neural Style Transfer 

<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/First ConvNet Artwork.png" />

### Summary

The goal of this project was to utilize the Neural Style Transfer (NST) algorithms to generate a unique artwork from two input images. The two images for input were the "content component" image and the "style component" image. The content component images is what the convolutional neural network (CNN) used as structure for the image generation. The style component image was then combined with the content image in several different layers of the CNN. Each layer of the CNN acts as a filter to extract certain features or attributes of each feature such as texture, corners, small shapes, or repeating color patterns. The lower level dimensionality features of the input images are filtered and learned in the first few layers of the CNN. These layers would be responsible for noticing color patterns and small repeating patterns. The higher level dimensionality features responsible for actual content creation such as buildings, and objects are filtered and learned in higher level layers of the CNN. Of course some of the layers responsible for both content and style overlap in the middle layers of the CNN, and this is where the NST actually takes place. 

The original arrangement of the photograph content is preserved, while the colors and minor structures that compose the style  become entangled. Therefore the generated image has a new unique style even though it shows the same content as the original photograph. 


### Model

Template code is provided in the `Generating Art Using Neural Style Transfer and VGG19.ipynb` notebook file. The VGG19 model consists of 19 layers and was constructed using Python 3 and Tensorflow in an iPyton Notebook. The input images were of shape 400px X 300px X 3 (rgb). The input images are used are displayed below:
<img width = "400" src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/louvre.jpg" />
<img width = "400" src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/sandstone.jpg" />

The VGG19 network used has already been pre-trained on a large number of images from the ImageNet database therefore it has learned the weights for low-level features and high-level features. The NST algorithm is developed by computing the cost of the content image, cost of the style image, and finally the cost of the generated image. The cost function of the style implements the Gram Matrix in the computation, also called the style matrix.  

The cost of the content is computed using the content image (C) and generated image (G) and their respective activation function outputs (a):

<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/Jcost_content.png" />

The Gram Matrix, is computed: 

<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/Gram_matrix.png" />

The cost of the style is computed using the style image (S) and generated image (G) and the mean squared error of the Gram Matrix:

<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/Jcost_style.png" />

Finally we compute the Total Cost using both the content cost and the style cost. This will be used with backpropagation and stochaistic gradient descent (SGD) to optimize the synthesis between the two images. 


<img src= "https://github.com/JeffGoodrich9791/Generating-Art-Using-Neural-Style-Transfer-and-VGG19/blob/master/Jtotal.png" />

The Total Cost is then set up and run in a Tensorflow session and optimized using Adam Optimization over the VGG19 network. 

### Run

> def model_nn(sess, input_image, num_iterations = 160):
    
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    sess.run(model['input'].assign(input_image))
    ### END CODE HERE ###
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        _ = sess.run(train_step)
        ### END CODE HERE ###
        
        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


### Results

The resulting output image after 160 iterations is displayed above the summary. Possibly better articulation of features could have been achieved using higher iterations in the 200+ number but I was satisfied with the style of the outcome using 160 iterations. 

