I would like to develop a web app to evaluate different Object Detection (ObjDet) and Segmentation (Seg) models. (Like Yolo, SAM2, mmdetection, rt-detr, etc...)

It should be possible to upload images to the server before any kind of processing and view them in the app.
Images should be organized in train/val/test folders in the app.
It should be possible to annotate images using a pre-trained model like SAM2. This will be later used to either train/fine-tune a model or used as visual prompt for model supporting this. Annotations should be saved in a database with a user tag.
It should be possible to run inference of ObjDet or Seg models on images. Generated annotations should be saved in a database with a model tag.
It should be possible to see the annotations produced by the inference, which should appear in a different color than annotations done by human in the training phase.
It should be possible to evaluate performance in terms of time execution and accuracy from the predictions.
It should be possible to select which model to be used for annotations and inferences.
Once some user annotations are available, it should be possible to train/fine-tune a model.
When launching the train/fine-tune of a model, it should be possible to monitor the execution of the pipeline in the app.
When a fine-tuned model has been trained, it should be possible to run inference on a batch of images and monitor the execution of the pipeline in the app.
