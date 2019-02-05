%% BPANN: Artificial Neural Network with Back Propagation
%% Author: RAjesh Amerineni
  function [weightCell, biasCell,layerOfNeurons,err]= BPANN(trainInp,trainOut,hiddenNeurons,errorThreshhold, iterations,learningRate)

    
   
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
    
   
    %---Print predictions
    fprintf('Ended with %d iterations.\n', iter);

 end
%% BackPropagate: Backpropagate the output through the network and adjust weights and biases

%% ForwardNetwork: Compute feed forward neural network, Return the output and output of each neuron in each layer
