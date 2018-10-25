function sigmoid(x){
    return 1 / (1 + Math.exp(-x));
}

function dSigmoid(y){
    return y * (1 - y);
}

class NeuralNetwork{
    constructor(inputNodes, hiddenNodes, outputNodes){
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        this.weightsIH = new Matrix(this.hiddenNodes, this.inputNodes);
        //console.table(this.weightsIH);
        this.weightsHO = new Matrix(this.outputNodes, this.hiddenNodes);
        this.weightsIH.randomise();
        //console.table(this.weightsIH);
        this.weightsHO.randomise();

        this.biasH = new Matrix(this.hiddenNodes, 1);
        this.biasO = new Matrix(this.outputNodes, 1);
        this.biasH.randomise();
        this.biasO.randomise();

        this.learningRate = 0.1;
    }

    feedForward(inputArray){
        let inputs = Matrix.fromArray(inputArray);
        let hidden = Matrix.multiply(this.weightsIH, inputs);

        // console.table(inputs);
        // console.table(this.weightsIH);

        hidden.add(this.biasH);
        hidden.map(sigmoid);

        // console.table(hidden);

        let output = Matrix.multiply(this.weightsHO, hidden);
        output.add(this.biasO);
        output.map(sigmoid);

        // console.table(output);

        return output.toArray();
    }

    train(inputArray, targetArray) {
        let inputs = Matrix.fromArray(inputArray);
        let hidden = Matrix.multiply(this.weightsIH, inputs);

        hidden.add(this.biasH);
        hidden.map(sigmoid);

        let outputs = Matrix.multiply(this.weightsHO, hidden);
        outputs.add(this.biasO);
        outputs.map(sigmoid);

        let targets = Matrix.fromArray(targetArray);
        let outputErrors = Matrix.subtract(targets, outputs);
             
        let gradients = Matrix.map(outputs, dSigmoid);
        gradients.multiply(outputErrors);
        gradients.multiply(this.learningRate);

        let hiddenT = Matrix.transpose(hidden);
        let weightHODeltas = Matrix.multiply(gradients, hiddenT);

        this.weightsHO.add(weightHODeltas);
        this.biasO.add(gradients);

        let whot = Matrix.transpose(this.weightsHO);
        let hiddenErrors = Matrix.multiply(whot, outputErrors);

        let hiddenGradient = Matrix.map(hidden, dSigmoid);
        hiddenGradient.multiply(hiddenErrors);
        hiddenGradient.multiply(this.learningRate);

        let inputsT = Matrix.transpose(inputs);
        let weightIHDeltas = Matrix.multiply(hiddenGradient, inputsT);
        
        this.weightsIH.add(weightIHDeltas);
        this.biasH.add(hiddenGradient);
    }
}