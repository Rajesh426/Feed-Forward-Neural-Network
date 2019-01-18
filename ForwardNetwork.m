function [realOutput, layerOutputCells] = ForwardNetwork(in, layer, weightCell, biasCell)
    layerCount = size(layer, 2);
    layerOutputCells = cell(1, layerCount);
    out = in;
    for layerIndex = 1:layerCount
        X = out;
        bias = biasCell{layerIndex};
        if layerIndex == layerCount
        % If want to use softmax activation function in the output layer uncomment the below line and then comment the next Sigmoid line
%              out = softmax((X * weightCell{layerIndex} + bias)');
                  out = Sigmoid((X * weightCell{layerIndex} + bias)');

             out = out';

        else
           out = Sigmoid(X * weightCell{layerIndex} + bias);
%              
        end
        layerOutputCells{layerIndex} = out;
   end
    realOutput = out;    
   
end
