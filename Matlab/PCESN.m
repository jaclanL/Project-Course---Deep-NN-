classdef PCESN
   properties
      nInputUnits, nReservoirUnits, nOutputUnits
      spectralRadius, sigma2, phi2, eta
      inputMean, inputAbs, M2, outputMean, t
      Win, Wres, Wself, Wfb, Wout, Wdir, Wtrain
      r, s, o, c, outFactor
      V
   end
   methods
      function obj = initPCESN(obj,nInputUnits, nReservoirUnits, nOutputUnits,...
                               spectralRadius, sigma2, phi2, outFactor)
         % Structure
         obj.nInputUnits = nInputUnits;
         obj.nReservoirUnits = nReservoirUnits;
         obj.nOutputUnits = nOutputUnits;

         % Parameters
         obj.spectralRadius = spectralRadius; % 0<sR<1 to ensure ESP! default 0.4???
         obj.sigma2 = sigma2;
         obj.phi2 = phi2;
         obj.eta = 1; % learning rate -  0.1 -> 0.01 eta^p < inf, sum(eta) = inf, 

         % Initalize input sum and mean variables
         obj.inputMean = zeros(nInputUnits,1);
         obj.inputAbs = 1*ones(nInputUnits,1); % small enough???
         obj.t = 0;

         % Initialize weights, H.Jaeger (Sparse reservoir weights)
         success = 0 ;                                               
         while success == 0
            % following block might fail, thus we repeat until we obtain a valid
            % ReservoirWeights matrix
            try
               ReservoirWeights = sprand(nReservoirUnits, nReservoirUnits, 1/nReservoirUnits);
               ReservoirWeights(ReservoirWeights ~= 0) = ReservoirWeights(ReservoirWeights ~= 0)  - 0.5;
               opts.Display = 0;
               maxVal = max(abs(eigs(ReservoirWeights,1, 'largestabs', opts)));
               ReservoirWeights = ReservoirWeights/maxVal;
               success = 1 ;
            catch
               success = 0 ; 
            end
        end
        obj.Wres = ReservoirWeights;
        obj.Wres = obj.Wres * obj.spectralRadius;
        
        obj.Win = eye(nInputUnits);
        obj.Wself = (2.0*rand(nReservoirUnits, nInputUnits)-1.0); % init not mentioned???
        obj.Wfb = (2.0 * rand(nReservoirUnits, nOutputUnits)- 1.0);
        obj.Wout = zeros(nOutputUnits,nReservoirUnits);
        obj.Wdir = zeros(nOutputUnits,nInputUnits);
        obj.Wtrain = [obj.Wout obj.Wdir]; % just a combination of [Wout Wdir]
        
        obj.r = zeros(nReservoirUnits,1);
        obj.s = zeros(nInputUnits,1);
        obj.o = zeros(nOutputUnits,1);
        obj.c = [obj.r;obj.s];
        
        obj.outFactor = outFactor;

        % Initialize covariance matrix
        obj.V = obj.phi2 * eye(nInputUnits + nReservoirUnits);
     end       
     function obj = trainESN(obj, inputSample,targetSample)
        % Normalize and center input online
        obj.inputMean = obj.inputMean + (inputSample-obj.inputMean)/(obj.t+1);
        inputSample = inputSample - obj.inputMean;

        % NOT PART OF ALGORITHM:
        obj.inputAbs = max(obj.inputAbs, abs(inputSample));
        inputSample = inputSample./obj.inputAbs;
        obj.t = obj.t + 1; % update time index

        obj.s = tanh(obj.Win * inputSample); % update self-organized layer
        obj.r = tanh(obj.Wres * obj.r + obj.Wself * obj.s + obj.Wfb * obj.o);
        ccurr = obj.c;
        obj.c = [obj.r;obj.s];
        
        % Simple smoothing (NOT PART OF ALGORITHM)
        oldOut = obj.o;
        obj.o = obj.outFactor*oldOut + (1-obj.outFactor)*(obj.Wtrain * obj.c);
        
        Vprev = obj.V;
        obj.V = inv(inv(Vprev) + (1/obj.sigma2) * (ccurr * ccurr'));
        a = obj.V/Vprev * obj.Wtrain';
        b = (1/obj.sigma2) * (obj.V * ccurr) * targetSample';

        obj.Wtrain = a'+b';

        % Calculate GHL update 
        dWin = obj.eta*(inputSample * obj.s' - tril(obj.s * obj.s') * obj.Win);
        obj.Win = obj.Win + dWin; % Update Win matrix

        % Update learning rate (NOT PART OF ALGORITHM)
        obj.eta = 1/sqrt(obj.t);
        
        % Welford's Online algorithm (NOT PART OF ALGORITHM)
        %oldM = obj.outputMean;
        %delta = targetSample-obj.outputMean;
        %obj.outputMean = obj.outputMean + delta/obj.t;
        %obj.M2 = obj.M2 + (targetSample-obj.outputMean) .* delta;
        %obj.sigma2 = [ones(obj.nReservoirUnits,1);repmat(obj.M2/obj.t,[3,1])];
     end
  end
end