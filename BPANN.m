%% BPANN: Artificial Neural Network with Back Propagation
%% Author: RAjesh Amerineni
  function [weightCell, biasCell,layerOfNeurons,err]= BPANN(trainInp,trainOut,hiddenNeurons,errorThreshhold, iterations,learningRate)
% function BPANN()
% %%%% input into network%%%
% [A,C,E,H,I,T,O,U]=creating_letters;
% output=[eye(6),eye(6),eye(6),eye(6),eye(6),eye(6),eye(6),eye(6),eye(6),eye(6)];
% output1=[eye(6),output,output,output,output,output];
% var=0.01;
% mean=0;
% % coloumn 
% noiseoutc=gaussiannoiseout(T,H,E,A,C,I,mean,var);
% noiseoutc1=gaussiannoiseout(T,H,E,A,C,I,mean,0.02);
% noiseoutc2=gaussiannoiseout(T,H,E,A,C,I,mean,0.03);
% noiseoutc3=gaussiannoiseout(T,H,E,A,C,I,mean,0.04);
% noiseoutc4=gaussiannoiseout(T,H,E,A,C,I,mean,0.05);
% reshape_CA=reshape(A,64,1);
% 
% 
% reshape_CC=reshape(C,64,1);
% 
% reshape_CE=reshape(E,64,1);
% 
% reshape_CH=reshape(H,64,1);
% 
% reshape_CI=reshape(I,64,1);
% 
% reshape_CT=reshape(T,64,1);
% input1=[reshape_CT,reshape_CH,reshape_CE,reshape_CA,reshape_CC,reshape_CI];
% % inout to neural network
% trainInp=[input1'];
%  testInp = trainInp;
% trainOut=eye(6);
%  testInp = trainInp;
%     testRealOut = trainOut;
% hiddenNeurons=[64 36];

% %%%% input assign end%%%%%%
    %---Set training parameters
%     iterations = 5000;
%     errorThreshhold = 0.001;
%     learningRate = 0.5;
    %---Set hidden layer type, for example: [4, 3, 2]
%     hiddenNeurons = [3 2];
    
    %---'Xor' training data
%     trainInp = [0 0; 0 1; 1 0; 1 1];
%     trainOut = [0; 1; 1; 0];
%     testInp = trainInp;
%     testRealOut = trainOut;
    
    % %---'And' training data
    % trainInp = [1 1; 1 0; 0 1; 0 0];
    % trainOut = [1; 0; 0; 0];
    % testInp = trainInp;
    % testRealOut = trainOut;
    assert(size(trainInp,1)==size(trainOut, 1),...
        'Counted different sets of input and output.');
    %---Initialize Network attributes
    inArgc = size(trainInp, 2);
    outArgc = size(trainOut, 2);
    trainsetCount = size(trainInp, 1);
    
    %---Add output layer
    layerOfNeurons = [hiddenNeurons, outArgc];
    layerCount = size(layerOfNeurons, 2);
    
    %---Weight and bias random range
    e = 1;
    b = -e;
    %---Set initial random weights
    weightCell = cell(1, layerCount);
    for i = 1:layerCount
        if i == 1
%             weightCell{1} = unifrnd(b, e, inArgc,layerOfNeurons(1));
              weightCell{1} = randn(inArgc,layerOfNeurons(1));
        else
%             weightCell{i} = unifrnd(b, e, layerOfNeurons(i-1),layerOfNeurons(i));
               weightCell{i} = randn(layerOfNeurons(i-1),layerOfNeurons(i));
        end
    end
    %---Set initial biases
    biasCell = cell(1, layerCount);
    for i = 1:layerCount
%         biasCell{i} = unifrnd(b, e, 1, layerOfNeurons(i));
              biasCell{i} = randn(1, layerOfNeurons(i));
    end
    %----------------------
    %---Begin training
    %----------------------
    for iter = 1:iterations
        for i = 1:trainsetCount
            % choice = randi([1 trainsetCount]);
            choice = i;
            sampleIn = trainInp(choice, :);
            sampleTarget = trainOut(choice, :);
            [realOutput, layerOutputCells] = ForwardNetwork(sampleIn, layerOfNeurons, weightCell, biasCell);
            [weightCell, biasCell] = BackPropagate(learningRate, sampleIn, realOutput, sampleTarget, layerOfNeurons, ...
                weightCell, biasCell, layerOutputCells);
        end
        %plot overall network error at end of each iteration
        error = zeros(trainsetCount, outArgc);
        for t = 1:trainsetCount
            [predict, layeroutput] = ForwardNetwork(trainInp(t, :), layerOfNeurons, weightCell, biasCell);
            p(t,:) = predict;
            error(t, : ) = predict - trainOut(t, :);
        end
            err(iter) = (sum(sum(power(error,2)'))/(trainsetCount*outArgc))^0.5;
%         err(iter)= (mean(-(sum(( trainOut.*log(p))'))));
 
%         figure(1);
%         plot(err);
        %---Stop if reach error threshold
        if err(iter) < errorThreshhold
            break;
        end
    end
    
    %--Test the trained network with a test set
%     testsetCount = size(testInp, 1);
%     error = zeros(testsetCount, outArgc);
%     for t = 1:testsetCount
%         [predict, layeroutput] = ForwardNetwork(testInp(t, :), layerOfNeurons, weightCell, biasCell);
%         p(t,:) = predict;
%         error(t, : ) = predict - testRealOut(t, :);
%     end
    %---Print predictions
    fprintf('Ended with %d iterations.\n', iter);
%     a = testInp;
%     b = testRealOut;
%     c = p';
%     x1_x2_act_pred_err = [a b c c-b]
    %---Plot Surface of network predictions
%     testInpx1 = [-1:0.1:1];
%     testInpx2 = [-1:0.1:1];
%     [X1, X2] = meshgrid(testInpx1, testInpx2);
%     testOutRows = size(X1, 1);
%     testOutCols = size(X1, 2);
%     testOut = zeros(testOutRows, testOutCols);
%     for row = [1:testOutRows]
%         for col = [1:testOutCols]
%             test = [X1(row, col), X2(row, col)];
%             [out, l] = ForwardNetwork(test, layerOfNeurons, weightCell, biasCell);
%             testOut(row, col) = out;
%         end
%     end
%     figure(2);
%     surf(X1, X2, testOut);
 end
%% BackPropagate: Backpropagate the output through the network and adjust weights and biases

%% ForwardNetwork: Compute feed forward neural network, Return the output and output of each neuron in each layer
