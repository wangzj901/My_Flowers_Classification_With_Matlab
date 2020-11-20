% Data set: https://ww2.mathworks.cn/help/deeplearning/ug/data-sets-for-deep-learning.html
% The conversion of raw data to jpg/png is also on the link.


% Reference: Matlab training courses: "Deep Learning with Matlab"
% https://matlabacademy.mathworks.com/



% Random seed: get same results 
rng(123)

% Location of the data set, here use matlab online.
images = '/MATLAB Drive/flowerPhotos';
imds = imageDatastore(images,'IncludeSubfolders',true,'LabelSource','foldernames');

% Summarize the number of images per category.
tbl = countEachLabel(imds)

% Determin the smallest amount of images in a category.
minSetCount = min(tbl{:,2})

% Reduce the num of images to 100. This will reduce the training time.
maxNumImages = 100;
minSetCount = min(maxNumImages,minSetCount);

% Trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Each set now has exactly the same number of images.
countEachLabel(imds)

% Split and load the datastore
[trainImgs,valImgs,testImgs] = splitEachLabel(imds,0.8,0.1,0.1,'randomized');

% Preprocessing the datastore. inputsize = [227 227].
augImdsTrain = augmentedImageDatastore([227 227],trainImgs);
augImdsVal = augmentedImageDatastore([227 227],valImgs);
augImdsTest = augmentedImageDatastore([227 227],testImgs);

% Load pretrained network "alexnet"
net = alexnet;

% Use the |fc7| layer of AlexNet for feature extraction.
layer = 'fc7';

% Get training features with activations function.
trainingFeatures = activations(net,augImdsTrain,layer,'OutputAs','rows');

% Train a classifier, using features and labels.
classifier = fitcecoc(trainingFeatures,trainImgs.Labels);

% Get test features with activations function.
testFeatures = activations(net,augImdsTest,layer,'OutputAs','rows');

% Make a prediction by the classfier model and test features.
testPred = predict(classifier,testFeatures);

% Get the accuracy.
nnz(testPred == testImgs.Labels)/numel(testPred)

% Get Confusion matrix.
[cmap,clabel] = confusionmat(testImgs.Labels,testPred);
heatmap(clabel,clabel,cmap)


% Show the layers of alexnet.
layers = net.Layers;

% Replace the 23th and 25th layer. 5 flowers in this set.
layers(end-2) = fullyConnectedLayer(5);
layers(end) = classificationLayer;

% Set the training Options.
options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001, ...
    'Plots','training-progress', ...
    'ValidationData',augImdsVal, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',15);

% Training the network.
transferNet = trainNetwork(augImdsTrain,layers,options);

% Use the net to classify the test set.
testPred = classify(transferNet,augImdsTest);

% Accuracy.
nnz(testPred == testImgs.Labels)/numel(testImgs.Labels)

% Confusion matrix.
[cmap,clabel] = confusionmat(testImgs.Labels,testPred);
heatmap(clabel,clabel,cmap)

% List the variable in workspace
whos net

% Accuracy = 80%.
