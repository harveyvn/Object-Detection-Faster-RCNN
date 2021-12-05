# Object Detection using Faster-RCNN in Tensorflow
This notebook demonstrates an example on how to use object detection models available in Tensorflow Hub. 
In the following sections, we can:
* Explore and download an available model on the Tensorflow Hub
* Preprocess an image for inference
* Run inference on the models and visualize the output

## Download the model from Tensorflow Hub and load the model
```
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = model.signatures['default']
```

As a result, we can run our detector and print the number of objects found followed by three lists:
* The detection scores of each object found (i.e. how confident the model is)
* The classes of each object found
* The bounding boxes of each object

```
# print results
print("Found %d objects." % len(prediction["detection_scores"]))

# draw predicted boxes over the image
image_with_boxes = draw_boxes(
                        image=sample, 
                        boxes=prediction["detection_boxes"],
                        class_names=prediction["detection_class_entities"],
                        scores=prediction["detection_scores"]
                    )
# display the image
display_image(image_with_boxes)
```
<p align="center"><img src="https://user-images.githubusercontent.com/3027146/144760327-461875d7-8472-4de2-9681-823485d54689.png" width="800"></p>
