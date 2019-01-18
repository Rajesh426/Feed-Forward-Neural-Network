clc
clear all
close all

    %---'Xor' training data
    trainInp = [0 0; 0 1; 1 0; 1 1];
    trainOut = [0; 1; 1; 0];
    testInp = trainInp;
    testRealOut = trainOut;
    
    % %---'And' training data
%     trainInp = [1 1; 1 0; 0 1; 0 0];
%     trainOut = [1; 0; 0; 0];
    testInp = trainInp;
    testRealOut = trainOut;
    assert(size(trainInp,1)==size(trainOut, 1),...
        'Counted different sets of input and output.');
    %---Initialize Network attributes
    inArgc = size(trainInp, 2);
    outArgc = size(trainOut, 2);
  

% Error Threshold for the neural Network Algorithm
errorThreshhold=0.01;
% Total Number of Iteration for the Neural Network to perform
iterations=10000;
% learning Rate of the neural Network
learningRate=0.05;

 % Number of neurons for input layer
 inArgc = size(trainInp, 2);
 % Number of neurons for output layer
 outArgc = size(trainOut, 2);

 % Number of neurons in the hidden layer
hiddenNeurons=[inArgc];
% training of neural netowrk BPANN call the Feedforward function for 
% Forward layer and call BackPropagation function for Backward error
% calculation
[weightCell, biasCell,layerOfNeurons]= BPANN(trainInp,trainOut,hiddenNeurons,errorThreshhold, iterations,learningRate)
% testing the traind network
 testsetCount = size(testInp, 1);
 % Error calculation for the Network
    error = zeros(testsetCount, outArgc);
    for t = 1:testsetCount
        [predict, layeroutput] = ForwardNetwork(testInp(t, :), layerOfNeurons, weightCell, biasCell);
        p(t,:) = predict;
        error(t, : ) = predict - testRealOut(t, :);
    end
    % prediction
   p
   % error
   error